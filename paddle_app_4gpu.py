import os
import gc
import json
import time
import uuid
import asyncio
import requests
import multiprocessing as mp
from multiprocessing.managers import SyncManager
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional
import base64

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from paddle_coordinates import coordinates_process, log_exception


# -------------------- Config --------------------
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("config.json not found. Provide configuration.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_data = json.load(f)

DEFAULT_DPI = config_data.get("default_dpi", 300)
CLIP_TOP_PIXELS = config_data.get("clip_top_pixels", 0)
DOWNLOAD_DIR = config_data.get("downloadDirectory", "downloads")
LOG_DIR = config_data.get("log_directory", "logs")
OCR_JSON_FOLDER = config_data.get("ocrJsonfolder", "ocr_json")

os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OCR_JSON_FOLDER, exist_ok=True)

# -------------------- Tunables --------------------
GPU_IDS        = [0, 1,2, 3]     # physical GPU ids to use â€” one worker per id
NUM_WORKERS    = len(GPU_IDS)  # parallel worker processes (one per GPU)
MAX_QUEUE_SIZE = 1000          # reject new jobs when this many are waiting
RESULT_TTL_SEC = 300           # seconds before a finished job record is reaped


class JobStatus:
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


# -------------------- Helpers --------------------
def process_logger(msg: str, logfile: str) -> None:
    try:
        line = f"[{datetime.now().isoformat()}] {msg}\n"
        with open(logfile, "a", encoding="utf-8") as fp:
            fp.write(line)
    except Exception as exc:
        log_exception(exc, "process_logger error:", logfile)


def clear_gpu_cache() -> None:
    """Free GPU memory after every job â€” runs inside the worker process."""
    gc.collect()
    try:
        import paddle
        paddle.device.cuda.empty_cache()
    except Exception:
        pass


def _safe_output_stem(job: dict, job_id: str) -> str:
    """Derive a filesystem-safe stem for the OCR output file, never raises."""
    file_name = job.get("FileName")
    if file_name:
        stem = os.path.splitext(file_name)[0]
        if stem:
            return stem
    return job_id


# def _make_job_dict(job_id: str, base64_data: Any, FileName: str, byteio: Any, pdfPath: str,
#                     per_job_log: Any, uuidName: str, estimated_pages: str
#                     ) -> dict:
#     return {
#         "job_id": job_id,
#         "base64": base64_data,
#         "FileName": FileName,
#         "byteio": byteio,
#         "pdfPath": pdfPath,  # will be set by worker after download
#         "per_job_log": per_job_log,
#         "uuidName": uuidName,
#         "pages": estimated_pages,
#         "status": JobStatus.QUEUED,
#         "result": None,
#         "error": None,
#         "created_at": time.time(),
#         "finished_at": None,
#     }

def _make_job_dict(
    job_id,
    fileId,
    caseSummaryId,
    documentDetailId,
    base64_data,
    FileName,
    env,
    request_type,
    byteio,
    pdfPath,
    per_job_log,
    uuidName,
    estimated_pages,
):
    return {
        "job_id": job_id,
        "fileId": fileId,
        "caseSummaryId": caseSummaryId,
        "documentDetailId": documentDetailId,
        "base64": base64_data,
        "FileName": FileName,
        "env": env,
        "request_type": request_type,
        "byteio": byteio,
        "pdfPath": pdfPath,
        "per_job_log": per_job_log,
        "uuidName": uuidName,
        "pages": estimated_pages,
        "status": JobStatus.QUEUED,
        "result": None,
        "error": None,
        "created_at": time.time(),
        "finished_at": None,
    }


# -------------------- Worker process (one per GPU) --------------------
def worker_process(worker_id: int, gpu_id: int, job_queue: mp.Queue,
                    job_registry: Any, stop_event: Any) -> None:
    # Scope CUDA visibility to exactly this GPU BEFORE importing paddle/paddleocr.
    # This must happen first in the child process â€” importing paddle at module
    # level in the parent, or setting CUDA_VISIBLE_DEVICES globally, causes all
    # workers to see a remapped/shared view of the GPUs instead of one each.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from paddleocr import PaddleOCR

    device_str = "gpu:0"  # always 0 â€” this process only ever sees ONE physical GPU

    print(f"[Worker-{worker_id}] Loading PaddleOCR on physical GPU {gpu_id} ({device_str}) â€¦", flush=True)
    pipeline = PaddleOCR(
        text_detection_model_name=    "PP-OCRv6_medium_det",
        text_recognition_model_name=  "PP-OCRv6_medium_rec",
        device=                       device_str,
        use_doc_orientation_classify= False,
        use_doc_unwarping=            False,
        use_textline_orientation=     False,
    )
    print(f"[Worker-{worker_id}] PaddleOCR ready on physical GPU {gpu_id}", flush=True)

    while not stop_event.is_set():
        # ---- Pull a job (1-second timeout so we can check stop_event) ----
        try:
            job_id = job_queue.get(timeout=1)
        except Exception:
            continue  # Queue.Empty or timeout -> loop back

        job = job_registry.get(job_id)
        if job is None:
            print(f"[Worker-{worker_id}] job_id {job_id} vanished, skipping")
            continue

        log_file = job.get("per_job_log") or os.path.join(LOG_DIR, f"{job_id}.log")

        # Everything from here down is wrapped so a bug in job setup can
        # NEVER kill the worker process â€” a dead worker silently stops
        # picking up jobs and every future job assigned to it hangs forever.
        try:
            output_stem = _safe_output_stem(job, job_id)
            output_file = os.path.join(OCR_JSON_FOLDER, f"{output_stem}.json")

            job["status"] = JobStatus.PROCESSING
            job_registry[job_id] = job  # write back â€” Manager dict needs this

            try:
                # -------- PDF SOURCE ----------
                folder = os.path.join(DOWNLOAD_DIR, job_id)
                os.makedirs(folder, exist_ok=True)

                # 1. DIRECT PDF PATH
                if job.get("pdfPath"):
                    pdf_path = job["pdfPath"]
                    print(f"[Worker-{worker_id}] using existing pdfPath: {pdf_path}")

                    if not os.path.isfile(pdf_path):
                        raise Exception(f"PDF path does not exist: {pdf_path}")

                # 2. DOWNLOAD FROM API
                elif job.get("documentDetailId"):

                    if job["env"].lower() == "stage":
                        base = config_data["stageBaseFileURL"]
                    elif job["env"].lower() == "demo":
                        base = config_data["demoBaseFileURL"]
                    else:
                        base = config_data["prodBaseFileURL"]

                    download_url = (base + config_data["downloadFileURL"]).format(
                        fileId=job["fileId"],
                        caseSummaryId=job["caseSummaryId"],
                        documentDetailId=job["documentDetailId"]
                    )

                    print(f"[Worker-{worker_id}] downloading from {download_url}")

                    pdf_path = downloadFile(
                        download_url,
                        job["FileName"],
                        log_file,
                        folder
                    )

                    print(f"[Worker-{worker_id}] downloaded to {pdf_path}")
                
                # 3. BASE64 FALLBACK
                elif job.get("base64"):
                    if not job.get("FileName"):
                        raise Exception("base64 provided but FileName is missing")

                    pdf_path = os.path.join(folder, job["FileName"])
                    with open(pdf_path, "wb") as f:
                        f.write(base64.b64decode(job["base64"]))

                    print(f"[Worker-{worker_id}] created pdf from base64: {pdf_path}")

                else:
                    raise Exception("No valid PDF source found (need pdfPath or base64)")

                print(f"[Worker-{worker_id}] starting OCR for {job_id} on {device_str}")
                process_logger(
                    f"[Worker-{worker_id}] START job={job_id} file={pdf_path} device={device_str}",
                    log_file,
                )

                ocr_result = coordinates_process(pipeline, pdf_path, log_file)

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(ocr_result, f, ensure_ascii=False, indent=4)

                job["result"] = {
                    "success": True,
                    "message": "OCR completed",
                    "json_file": output_file,
                    "ocr_result": ocr_result,
                }
                job["status"] = JobStatus.DONE
                process_logger(f"[Worker-{worker_id}] DONE job={job_id}", log_file)

            except Exception as exc:
                log_exception(exc, f"worker-{worker_id}", log_file)
                job["error"] = str(exc)
                job["result"] = {"success": False, "error": str(exc)}
                job["status"] = JobStatus.FAILED
                process_logger(f"[Worker-{worker_id}] FAIL job={job_id} error={exc}", log_file)

            finally:
                job["finished_at"] = time.time()
                job_registry[job_id] = job  # write back final state

                process_logger(f"[Worker-{worker_id}] Clearing GPU cache after job={job_id}", log_file)
                clear_gpu_cache()

                print(f"[Worker-{worker_id}] finished job={job_id}  status={job['status']}", flush=True)

        except Exception as outer_exc:
            # Last-resort catch: something failed before/around the inner
            # try block (e.g. a malformed job dict). Mark the job FAILED
            # instead of leaving it stuck at QUEUED, and keep the worker alive.
            print(f"[Worker-{worker_id}] UNEXPECTED ERROR on job={job_id}: {outer_exc}", flush=True)
            process_logger(f"[Worker-{worker_id}] UNEXPECTED ERROR job={job_id} error={outer_exc}", log_file)
            try:
                job["status"] = JobStatus.FAILED
                job["error"] = str(outer_exc)
                job["result"] = {"success": False, "error": str(outer_exc)}
                job["finished_at"] = time.time()
                job_registry[job_id] = job
            except Exception:
                pass
            clear_gpu_cache()

    print(f"[Worker-{worker_id}] stop signal received, exiting", flush=True)


# -------------------- Reaper (asyncio task in the FastAPI process) --------------------
async def result_reaper(job_registry: Any) -> None:
    """Remove finished job records older than RESULT_TTL_SEC."""
    while True:
        await asyncio.sleep(120)
        now = time.time()
        expired = [
            jid for jid, job in list(job_registry.items())
            if job.get("finished_at") and (now - job["finished_at"]) > RESULT_TTL_SEC
        ]
        for jid in expired:
            try:
                del job_registry[jid]
            except KeyError:
                pass
        if expired:
            print(f"[Reaper] Removed {len(expired)} expired job(s)")


# -------------------- Application lifespan --------------------
# Module-level handles so endpoints can access them
_manager: Optional[SyncManager] = None
_job_queue: Optional[mp.Queue] = None
_job_registry: Optional[Any] = None
_stop_event: Optional[Any] = None
_workers: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager, _job_queue, _job_registry, _stop_event, _workers

    # ---- Start the Manager server process ----
    _manager = mp.Manager()
    _job_queue = _manager.Queue(maxsize=MAX_QUEUE_SIZE)
    _job_registry = _manager.dict()
    _stop_event = _manager.Event()

    # ---- Spawn worker processes â€” one per GPU ----
    for i, gpu in enumerate(GPU_IDS):
        p = mp.Process(
            target=worker_process,
            args=(i, gpu, _job_queue, _job_registry, _stop_event),
            daemon=True,
            name=f"ocr-worker-{i}-gpu{gpu}",
        )
        p.start()
        _workers.append(p)
        print(f"[Main] Spawned worker process {p.pid} (worker-{i}, gpu-{gpu})")

    # ---- Start the reaper asyncio task ----
    reaper_task = asyncio.create_task(result_reaper(_job_registry), name="result-reaper")

    yield  # FastAPI serves requests here

    # ---- Graceful shutdown ----
    print("[Main] Sending stop signal to workers â€¦")
    _stop_event.set()

    for p in _workers:
        p.join(timeout=30)
        if p.is_alive():
            print(f"[Main] Force-killing worker {p.pid}")
            p.kill()

    reaper_task.cancel()
    await asyncio.gather(reaper_task, return_exceptions=True)

    _manager.shutdown()
    print("[Main] Shutdown complete")


# -------------------- Download helper --------------------
def downloadFile(url: str, local_filename: str, logfile: str, folder: str) -> Optional[str]:
    try:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, local_filename)
        process_logger(f"Downloading {url} -> {filepath}", logfile)
        with requests.get(url, stream=True, timeout=1200) as r:
            r.raise_for_status()
            with open(filepath, "wb") as out:
                for chunk in r.iter_content(chunk_size=8192):
                    out.write(chunk)
        process_logger(f"Downloaded: {filepath}", logfile)
        return filepath
    except Exception as e:
        log_exception(e,"downloadFile", logfile)
        return None


# -------------------- FastAPI app --------------------
app = FastAPI(lifespan=lifespan, title="PaddleOCR API")


@app.get("/")
def health():
    q_size = 0
    try:
        q_size = _job_queue.qsize()
    except Exception:
        pass

    worker_alive = [p.is_alive() for p in _workers]

    return {
        "status": "running",
        "model": "PaddleOCR-V6",
        "workers": NUM_WORKERS,
        "workers_alive": worker_alive,
        "gpus": GPU_IDS,
        "queued": q_size,
    }


class Item(BaseModel):
    fileId : str
    caseSummaryId : str
    documentDetailId : str
    base64: str | None = None
    FileName: str | None = None
    pdfPath: str | None = None  # will be set by worker after download
    pages: str | None = None
    env : str
    request_type : str | None = None


@app.post("/", status_code=202)
async def submit_ocr(req: Item):
    """
    Submit a file for OCR.
    Returns a job_id immediately (HTTP 202).
    Poll  GET /ocr/status/{job_id}
    or block on  GET /ocr/wait/{job_id}.
    """
    try:
        job_id = str(uuid.uuid4())
        uuidName = job_id
        fileId = req.fileId
        pdfPath = req.pdfPath
        # print("pdfPath : ", pdfPath)
        caseSummaryId = req.caseSummaryId
        documentDetailId = req.documentDetailId
        base64_data = req.base64
        FileName = req.FileName
        env = req.env
        request_type = req.request_type
        byteio = req.base64

        if not FileName and not pdfPath:
            raise HTTPException(status_code=422, detail="FileName is required (used for logging/output naming).")

        safe_name = FileName if FileName else os.path.basename(pdfPath or job_id)
        per_job_log = os.path.join(LOG_DIR, f"{uuidName}_{safe_name}.log")

        estimated_pages = req.pages

        try:
            q_size = _job_queue.qsize()
        except Exception:
            q_size = 0

        if q_size >= MAX_QUEUE_SIZE:
            raise HTTPException(status_code=503, detail="Queue is full. Try again later.")

        job = _make_job_dict(job_id,fileId, caseSummaryId, 
                        documentDetailId, base64_data, FileName,
                        env,request_type, byteio, pdfPath,
                        per_job_log, uuidName, estimated_pages)
        _job_registry[job_id] = job     # register before enqueue so workers see it
        _job_queue.put(job_id)          # send only the id â€” registry holds the data

        print(f"Received job {job_id}")

        return {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "queue_size": _job_queue.qsize(),
            "message": f"Job queued. Poll /ocr/status/{job_id}.",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API ERROR] {e}")
        return {
            "job_id": "",
            "status": "Error",
            "queue_size": 0,
            "message": "Error in QUEUE",
        }


@app.get("/ocr/status/{job_id}")
def job_status(job_id: str):
    """Non-blocking status check."""
    job = _job_registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    resp: Dict[str, Any] = {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job["created_at"],
    }
    if job["finished_at"]:
        resp["finished_at"] = job["finished_at"]
        resp["duration_sec"] = round(job["finished_at"] - job["created_at"], 2)
    if job["status"] in (JobStatus.DONE, JobStatus.FAILED):
        resp["result"] = job["result"]
    return resp


@app.get("/ocr/wait/{job_id}")
async def wait_for_job(job_id: str, timeout: float = 300.0):
    """
    Long-poll: blocks until the job finishes or timeout seconds elapse.
    Useful for synchronous callers that want a single round-trip.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _job_registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        if job["status"] in (JobStatus.DONE, JobStatus.FAILED):
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "duration_sec": round(job["finished_at"] - job["created_at"], 2),
                "result": job["result"],
            }
        await asyncio.sleep(0.5)  # poll every 500 ms without burning CPU

    raise HTTPException(
        status_code=408,
        detail=f"Job '{job_id}' did not complete within {timeout}s.",
    )


@app.get("/ocr/queue")
def queue_info():
    """Live snapshot of the queue and all tracked jobs."""
    try:
        q_size = _job_queue.qsize()
    except Exception:
        q_size = -1

    return {
        "queued": q_size,
        "max_queue": MAX_QUEUE_SIZE,
        "workers": NUM_WORKERS,
        "gpus": GPU_IDS,
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "created_at": j["created_at"],
            }
            for j in _job_registry.values()
        ],
    }


if __name__ == "__main__":
    # "spawn" is required on Linux with CUDA to avoid forking GPU state
    mp.set_start_method("spawn", force=True)
    uvicorn.run(app, host="0.0.0.0", port=9005, reload=False)


import fitz
import gc
import json
import os
import re
import shutil
import sys
import time
import torch
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

import easyocr
import numpy as np
import redis
import requests

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
GPU_COUNT = 3
QUEUE_PREFIX = "ocr_queue_gpu_"
counter_key = "gpu_rr_counter"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OCR_JSON_FOLDER, exist_ok=True)


# -------------------- Logging helpers --------------------
def nowstr() -> str:
    return datetime.now().isoformat()


def processLogger(msg: str, logfile: str) -> None:
    try:
        s = f"[{nowstr()}] {msg}\n"
        with open(logfile, "a", encoding="utf-8") as fp:
            fp.write(s)
    except Exception:
        print(f"processLogger failed: {msg}", file=sys.stderr)


def log_exception(module: str, logfile: str, exc: Optional[BaseException] = None) -> None:
    exc_type, exc_obj, tb = sys.exc_info()
    log_date = nowstr()
    lineno = tb.tb_lineno if tb else "N/A"
    ob = f"\nTime - {log_date} -->> ERROR IN {module} -->> LINE.NO-{lineno} : {exc_obj}\n"
    ob += traceback.format_exc()
    print(ob)
    try:
        with open(logfile, "a", encoding="utf-8") as fp:
            fp.writelines(ob)
    except Exception:
        print("Failed to write exception to log file", logfile, file=sys.stderr)


def safe_delete_path(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f"safe_delete_path failed for {path}: {e}", file=sys.stderr)


r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

QUEUE_NAME = "ocr_queue"


# -------------------- Download helper --------------------
def downloadFile(url: str, local_filename: str, logfile: str, folder: str) -> Optional[str]:
    try:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, local_filename)
        processLogger(f"Downloading {url} -> {filepath}", logfile)
        with requests.get(url, stream=True, timeout=1200) as r:
            r.raise_for_status()
            with open(filepath, "wb") as out:
                for chunk in r.iter_content(chunk_size=8192):
                    out.write(chunk)
        processLogger(f"Downloaded: {filepath}", logfile)
        return filepath
    except Exception as e:
        log_exception("downloadFile", logfile, e)
        return None


def ocr_coordinates(result, page_num, width, height, dpi, logfile):
    try:
        all_res = []
        for i, entry in enumerate(result):
            en_text = entry['text'].strip()
            bbox = entry['boxes']
            score_ = float(entry["confident"])
            x0 = float(bbox[0][0])
            y0 = float(bbox[0][1])
            x1 = float(bbox[2][0])
            y1 = float(bbox[2][1])
            all_res.append({"x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                            'width': width,
                            'height': height,
                            'text': en_text,
                            'Page': page_num,
                            'confident_score': score_})
        all_res = sorted(all_res, key=lambda x: x["y0"])

        # Build blank fillers + tokens row-wise to preserve spacing
        all_coordinates = []
        for i, data in enumerate(all_res):
            x0 = data["x0"]
            y0 = data["y0"]
            x1 = data["x1"]
            y1 = data["y1"]
            en_text = data["text"]

            last_y0 = y0 - float(all_res[i - 1]["y0"])

            if not i == 0:
                last_y0 = y0 - float(all_res[i - 1]["y0"])
                last_x1 = float(all_res[i - 1]["x1"])

                if last_y0 > 10:
                    x1_ = float(all_res[i - 1]["x1"])
                    all_coordinates.append({
                        "x0": x1_,
                        "y0": y0,
                        "x1": width,
                        "y1": y1,
                        'width': width,
                        'height': height,
                        'text': "",
                        'Page': data["Page"],
                        'confident_score': 0
                    })

                    all_coordinates.append({
                        "x0": 0,
                        "y0": y0,
                        "x1": x0,
                        "y1": y1,
                        'width': width,
                        'height': height,
                        'text': "",
                        'Page': data["Page"],
                        'confident_score': 0
                    })
                    all_coordinates.append({
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        'width': width,
                        'height': height,
                        'text': en_text,
                        'Page': data["Page"],
                        'confident_score': data["confident_score"]
                    })
                else:

                    x1_ = float(all_res[i - 1]["x1"])
                    if x0 > x1_:
                        all_coordinates.append({
                            "x0": x1_,
                            "y0": y0,
                            "x1": x0,
                            "y1": y1,
                            'width': width,
                            'height': height,
                            'text': "",
                            'Page': data["Page"],
                            'confident_score': 0
                        })

                    all_coordinates.append({
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        'width': width,
                        'height': height,
                        'text': en_text,
                        'Page': data["Page"],
                        'confident_score': data["confident_score"]
                    })
            else:

                all_coordinates.append({
                    "x0": 0,
                    "y0": y0,
                    "x1": x0,
                    "y1": y1,
                    'width': width,
                    'height': height,
                    'text': "",
                    'Page': data["Page"],
                    'confident_score': 0
                })
                all_coordinates.append({
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    'width': width,
                    'height': height,
                    'text': en_text,
                    'Page': data["Page"],
                    'confident_score': data["confident_score"]
                })
        coordinates = []
        for i, data in enumerate(all_coordinates):
            x0 = data["x0"]
            y0 = data["y0"]
            x1 = data["x1"]
            y1 = data["y1"]
            x0_ = x0 * 72 / dpi
            y0_ = y0 * 72 / dpi
            x1_ = x1 * 72 / dpi
            y1_ = y1 * 72 / dpi
            data["x0"] = x0_
            data["y0"] = y0_
            data["x1"] = x1_
            data["y1"] = y1_
            data["width"] = round(width * 72 / dpi)
            data["height"] = round(height * 72 / dpi)
            coordinates.append(data)
        return coordinates
    except Exception as e:
        log_exception("Coordinates", logfile, e)


def Text_easy_ocr_spacing(ocr_results, page_num, page_width_px, page_height_px, dpi, logfile):
    try:
        import numpy as np
        y_tolerance_factor = 0.6
        coordinates = ocr_coordinates(ocr_results, page_num, page_width_px, page_height_px, dpi, logfile)
        if not ocr_results:
            return ""

        # ------------ Normalize OCR words ---------------
        words = []
        for obj in ocr_results:
            text = obj["text"].strip()
            if not text:
                continue

            box = obj["boxes"]

            # Force numeric ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ float
            x1 = float(box[0][0])
            y1 = float(box[0][1])
            x3 = float(box[2][0])
            y3 = float(box[2][1])

            width = float(abs(float(box[1][0]) - float(box[0][0])))
            height = float(abs(float(box[2][1]) - float(box[1][1])))

            center_y = (y1 + y3) / 2.0

            words.append({
                "text": text,
                "x1": float(x1),
                "x2": float(x3),
                "y": float(center_y),
                "width": float(width),
                "height": float(height)
            })

        # ------------ Determine median text size ----------
        median_height = float(np.median([w["height"] for w in words]))
        median_width = float(np.median([w["width"] for w in words]))

        # ------------ ROW-FIRST GROUPING ------------------
        words = sorted(words, key=lambda w: w["y"])  # vertical sort

        rows = []
        current_row_y = None
        current_row = []

        for w in words:
            if current_row_y is None or abs(w["y"] - current_row_y) > (median_height * y_tolerance_factor):
                if current_row:
                    rows.append(current_row)
                current_row = [w]
                current_row_y = w["y"]
            else:
                current_row.append(w)

        if current_row:
            rows.append(current_row)

        # ------------ SORT WORDS WITHIN ROW BY X ----------
        for r in rows:
            r.sort(key=lambda w: w["x1"])

        # ------------ RECONSTRUCT LINES -------------------
        reconstructed_lines = []
        for r in rows:
            line = []
            last_x = None
            for w in r:
                if last_x is not None:
                    gap = float(w["x1"] - last_x)

                    # convert EVERYTHING to python int safely
                    safe_gap = gap / max(1.0, median_width * 0.8)
                    safe_gap = int(max(1, round(safe_gap)))

                    line.append(" " * safe_gap)

                line.append(w["text"])
                last_x = float(w["x2"])

            reconstructed_lines.append("".join(line))
        return "\n".join(reconstructed_lines), coordinates
    except Exception as e:
        log_exception("Text_easy_ocr_spacing", logfile, e)
        return ""


def ocr_process_pdf(pdf_path: str, logfile: str, reader, dpi: int = DEFAULT_DPI, clip_top: int = CLIP_TOP_PIXELS) -> \
        Dict[str, Any]:
    """
    Process a PDF using a pre-initialized reader. Returns a result dict.
    """
    print(f"Starting OCR for {pdf_path}")
    start_time = time.time()
    result_json = {
        "texts": [],
        "coordinates": [],
        "confidence": [],
        "total_pages": 0,
        "processing_time_seconds": 0.0,
        "error": ""
    }

    try:
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
        except Exception as e:
            log_exception("Error opening PDF", logfile)
            result_json["error"] = e
            return result_json

        texts = []
        coords = []
        confs = []
        doc = fitz.open(pdf_path)
        for page_num in range(1, total_pages + 1):
            try:
                page = doc.load_page(page_num - 1)
                # page_text = page.get_text()
                # clean_text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', '', page_text)
                # if len(page_text.split(' '))<100:
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                page_width_px = pix.w
                page_height_px = pix.h

                try:
                    result = reader.readtext(img, output_format="dict", paragraph=False)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or isinstance(e, torch.cuda.OutOfMemoryError):
                        processLogger(f"CUDA OOM on page {page_num}", logfile)
                        try:
                            if total_pages > 10 and page_num % 10 == 0:
                                torch.cuda.empty_cache()
                        except:
                            pass
                        result_json["error"] = f"CUDA OOM on page {page_num}"
                        return result_json
                    else:
                        raise

                confs_page = []
                for r in result:
                    c = r.get("confident") or r.get("confidence")
                    if c is not None:
                        try:
                            confs_page.append(float(c))
                        except:
                            pass
                avg_conf = round(float(np.mean(confs_page)), 4) if confs_page else 0.0

                # text, coordinates = Text_easy_ocr_spacing(result, page_num, page_width_px, logfile, dpi=dpi, clip_top_pixels=clip_top)
                text, coordinates = Text_easy_ocr_spacing(result, page_num, page_width_px, page_height_px, dpi, logfile)
                texts.append(text or "")
                coords.append(coordinates or [])
                confs.append(avg_conf)

                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                # else:
                #     texts.append(page_text or "")
                #     coords.append([])
                #     confs.append(0.75)
            except Exception:
                log_exception(f"OCR ERROR on page {page_num}", logfile)
                texts.append("")
                coords.append([])
                confs.append(0.0)
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            finally:
                # IMPORTANT: delete img BEFORE pix, then delete page
                try:
                    if img is not None:
                        del img
                    if pix is not None:
                        del pix
                    if page is not None:
                        del page
                except Exception:
                    pass

                # force immediate cleanup
                try:
                    gc.collect()
                except:
                    pass
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
        doc.close()
        processing_time = time.time() - start_time
        result_json.update({
            "texts": texts,
            "coordinates": coords,
            "confidence": confs,
            "total_pages": total_pages,
            "processing_time_seconds": round(processing_time, 2),
            "error": ""
        })
        return result_json

    except Exception:
        log_exception("ocr_worker_process", logfile)
        result_json["error"] = "Internal OCR worker failure"
        return result_json


def ocr_process_pdf_with_fitz(pdf_path: str, logfile: str, reader, dpi: int = DEFAULT_DPI,
                              clip_top: int = CLIP_TOP_PIXELS) -> Dict[str, Any]:
    """
    Leak-aware OCR processing for a PDF using PyMuPDF + EasyOCR.
    """
    start_time = time.time()
    result_json = {
        "texts": [],
        "coordinates": [],
        "confidence": [],
        "total_pages": 0,
        "processing_time_seconds": 0.0,
        "error": ""
    }

    try:
        # OPEN PDF ONCE
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            result_json["total_pages"] = total_pages
        except Exception:
            log_exception("Error opening PDF", logfile)
            result_json["error"] = "Error opening PDF"
            return result_json

        # REMOVE ANNOTATIONS ONCE
        for i in range(total_pages):
            p = doc.load_page(i)
            a = p.first_annot
            while a:
                nxt = a.next
                try:
                    p.delete_annot(a)
                except Exception:
                    # ignore deletion errors for safety
                    pass
                a = nxt
            # free page object reference
            del p

        texts = []
        coords = []
        confs = []

        # PROCESS PAGES
        for page_num in range(total_pages):
            page = None
            pix = None
            img = None
            result = None

            try:
                page = doc.load_page(page_num)

                # ------------------ TEXT EXTRACTION VIA PDF ------------------
                page_text = page.get_text() or ""
                page_text = page_text.strip()
                page_text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', '', page_text)

                # -------------------------------------------------------------
                # CASE 1: PDF text exists ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ DO NOT OCR, DO NOT RENDER PAGE
                # -------------------------------------------------------------
                if page_text:
                    texts.append(page_text)
                    coords.append([])
                    confs.append(0.85)

                    # Release early
                    del page
                    continue

                # -------------------------------------------------------------
                # CASE 2: No PDF text ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ Render image and OCR
                # -------------------------------------------------------------
                # Render to image ONLY in this path
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=mat)

                # Convert pixmap to numpy
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                page_width_px = pix.w
                page_height_px = pix.h

                # Run OCR
                try:
                    result = reader.readtext(img, output_format="dict", paragraph=False)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or isinstance(e, torch.cuda.OutOfMemoryError):
                        processLogger(f"CUDA OOM on page {page_num + 1}", logfile)
                        try:
                            torch.cuda.empty_cache()
                        except:
                            pass
                        result_json["error"] = f"CUDA OOM on page {page_num + 1}"
                        return result_json
                    else:
                        raise

                # Compute average confidence
                confs_page = []
                for r in result:
                    c = r.get("confident") or r.get("confidence")
                    if c is not None:
                        try:
                            confs_page.append(float(c))
                        except:
                            pass
                avg_conf = round(float(np.mean(confs_page)), 4) if confs_page else 0.0

                # Normalize text and coordinates
                text, coordinates = Text_easy_ocr_spacing(result, page_num + 1, page_width_px, page_height_px, dpi,
                                                          logfile)

                texts.append(text or "")
                coords.append(coordinates or [])
                confs.append(avg_conf)

            except Exception:
                log_exception(f"OCR ERROR ON PAGE {page_num + 1}", logfile)
                texts.append("")
                coords.append([])
                confs.append(0.0)

            finally:
                # IMPORTANT: delete img BEFORE pix, then delete page
                try:
                    if img is not None:
                        del img
                    if pix is not None:
                        del pix
                    if page is not None:
                        del page
                except Exception:
                    pass

                # force immediate cleanup
                try:
                    gc.collect()
                except:
                    pass
                try:
                    torch.cuda.empty_cache()
                except:
                    pass

        # FINALIZE
        processing_time = round(time.time() - start_time, 2)
        result_json.update({
            "texts": texts,
            "coordinates": coords,
            "confidence": confs,
            "processing_time_seconds": processing_time,
            "error": ""
        })

        doc.close()
        return result_json

    except Exception:
        log_exception("ocr_worker_process", logfile)
        result_json["error"] = "Internal OCR worker failure"
        try:
            doc.close()
        except:
            pass
        return result_json


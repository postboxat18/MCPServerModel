
import json
import os
import uuid
from datetime import datetime
import sys
import time


os.makedirs("outputs", exist_ok=True)


def processLogger(msg, logfile):
    try:
        s = f"[{datetime.now().isoformat()}] {msg}\n"
        with open(logfile, 'a', encoding='utf-8') as fp:
            fp.write(s)
    except Exception as e:
        print("processLogger failed:", e)


def log_exception(e, func_name, logfile):
    try:
        exc_type, exc_obj, tb = sys.exc_info()
        lineno = tb.tb_lineno if tb else "N/A"
        error_message = f"\n[{datetime.now()}] In {func_name} LINE.NO-{lineno} : {exc_obj} error {e}"
        print(error_message)
        with open(logfile, 'a', encoding='utf-8') as fp:
            fp.writelines(error_message + "\n")
    except Exception as ee:
        print("Logging failed:", ee)


def poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]

    x1 = min(xs)/2
    y1 = min(ys)/2
    x2 = max(xs)/2
    y2 = max(ys)/2

    return [x1, y1, x2, y2], x1, y1, x2, y2

def is_inside(inner, outer):
    return (
        inner[0] >= outer[0] and
        inner[1] >= outer[1] and
        inner[2] <= outer[2] and
        inner[3] <= outer[3]
    )

def paddle_prediction(pipeline, pdf_path, output_path):
    try:
        print("Starting OCR process...")
        # print(pdf_path)
        output = pipeline.predict(input=pdf_path,
                                #   use_layout_detection=False,
                                #     prompt_label="spotting"
                                    )
        # files_ = os.listdir()

        # Assuming 'output' holds the output logs from pipeline.predict()
        for page_idx, res in enumerate(output):
            # res.print()
            res.save_to_json(save_path=output_path)

        return True
    except Exception as e:
        print("paddle prediction error",e)


def coordinates_process(pipeline, pdf_path,log_file):
    merged_blocks = {}

    merged_blocks["texts"] = []
    merged_blocks["total_pages"] = 0
    merged_blocks["processing_time_seconds"] = "00.00"
    merged_blocks["coordinates"] = []
    merged_blocks["confidence"] = 0.0
    
    file_name = "pdf_datas"

    try:  
        uuid_name = str(uuid.uuid4())
        ocr_data_path = os.path.join("outputs", f'{file_name}_{uuid_name}')
        os.makedirs(ocr_data_path, exist_ok=True)
        st_time = time.time()
        processLogger(f"Paddle ocr Started >> {st_time}", log_file)
        Is_file_saved = paddle_prediction(pipeline, pdf_path, ocr_data_path)
        en_time = time.time()
        overall_process_time = en_time-st_time
        merged_blocks["processing_time_seconds"] =  round(overall_process_time, 2)

        processLogger(f"Paddle ocr Completed >> {overall_process_time}", log_file)

        if not Is_file_saved:
            return merged_blocks
    
        files_ = os.listdir(ocr_data_path)
        sorted_files= sorted(files_, key=lambda x: int(x.split("_res")[0].split("_")[-1]))

        merged_blocks["total_pages"] = len(files_)
        page_avg_confidence = []

        for page_indx, file_name in enumerate(sorted_files):
            output_path = os.path.join(ocr_data_path, file_name)
            with open(output_path, "r") as rd:
                data = json.load(rd)

            polys = data.get("rec_polys", "")
            texts = data.get("rec_texts", "")
            text_confidence = data.get("rec_scores", [])
            page_confidence_average = sum(text_confidence) / len(text_confidence) if text_confidence else 0
            page_avg_confidence.append(page_confidence_average)

            # for block in data["parsing_res_list"]:
            # block_bbox = block["block_bbox"]
            # block["page_width"] = data.get("width", 612)
            # block["page_height"] = data("height", 792)

            block_words = []
            coordinate_data = []
            page_text = ""

            for poly, text, conf in zip(polys, texts, text_confidence):
                if not page_text:
                    page_text = text
                else:
                    page_text = page_text + "\n" + text
                word_bbox,x0, y0, x1, y1 = poly_to_bbox(poly)
                # print(word_bbox)
                # if is_inside(word_bbox, block_bbox):
                coordinate_data.append({
                                        "Page": page_indx+1,
                                        "confident_score": conf,
                                        "height": data.get("height", 792),
                                        "text": text,
                                        "width": data.get("width", 612),
                                        "x0": x0,
                                        "x1": x1,
                                        "y0": y0,
                                        "y1": y1
                                        })

            merged_blocks["coordinates"].append(coordinate_data)
            merged_blocks["texts"].append(page_text)
            
            rd.close()
            os.remove(output_path)

        merged_blocks["confidence"] = page_avg_confidence
        coordinate_path = os.path.join(ocr_data_path, "Output_coordinates.json")
        # with open(coordinate_path, "w") as wt:
        #     json.dump(merged_blocks, wt, indent=4)
    except Exception as e:
        log_exception(e, "coordinates_process", log_file)

    
    return merged_blocks


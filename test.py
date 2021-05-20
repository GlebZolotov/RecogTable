import cv2 as cv
import numpy as np
import sys
import math
from typing import List, Iterator, Tuple, Optional
import fitz
import pytesseract
import os


def rotation(image: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate input image to angle with adding white border in output image"""
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv.getRotationMatrix2D(img_c, angle_in_degrees, 1)

    rad = math.radians(angle_in_degrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    out_img = cv.warpAffine(image, rot, (b_w, b_h), flags=cv.INTER_LINEAR, borderValue=(255, 255, 255))
    return out_img


def find_templ(in_img: np.ndarray, templ: np.ndarray, thr: float = 0.05) -> List[Tuple[int, int]]:
    """Find template with mask in input image and return the list of founded points"""
    mask = cv.threshold(templ, 160, 255, cv.THRESH_BINARY_INV)[1]
    if in_img.shape[0] <= templ.shape[0] or in_img.shape[1] <= templ.shape[1]:
        return []
    result = cv.matchTemplate(in_img, templ, cv.TM_SQDIFF, None, mask)
    cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    if math.fabs(min_val - max_val) < 0.000001:
        return []
    min_thresh = min_val + thr
    match_locations = np.where(result <= min_thresh)

    w, h = templ.shape[::-1]
    w = w // 2
    h = h // 2
    res_points = [(x + w, y + h) for (x, y) in zip(match_locations[1], match_locations[0])]

    for i_ in range(len(res_points)):
        for j_ in range(len(res_points) - 1, i_, -1):
            if (res_points[i_][0] - res_points[j_][0]) ** 2 + (res_points[i_][1] - res_points[j_][1]) ** 2 < 100:
                del res_points[j_]

    return res_points


def quad_of_point(shape: List[int], p: List[int]) -> int:
    """Return number of fourth where is point p, 0 is right-up, 1 is left-down, 2 - left-up"""
    if p[0] > shape[0] // 2:
        if p[1] <= shape[1] // 2:
            return 0
        else:
            return 3
    else:
        if p[1] <= shape[1] // 2:
            return 2
        else:
            return 1


def quad_of_table(shape: List[int], edges: List[int]) -> int:
    """Return the fourth where are points of edges"""
    return quad_of_point(shape, [edges[0], edges[2]])


def recog_text_full(in_img: np.ndarray) -> str:
    """Recognition of russian text in input image"""
    pytesseract.pytesseract.tesseract_cmd = "Tesseract-OCR\\tesseract.exe"
    text = pytesseract.image_to_string(in_img, lang='rus')
    return text.strip()


def recog_numb_full(in_img: np.ndarray) -> str:
    """Recognition of numbers in input image"""
    pytesseract.pytesseract.tesseract_cmd = "Tesseract-OCR\\tesseract.exe"
    text = pytesseract.image_to_string(in_img, config='digits')
    return text.strip()


def load_file_using_fitz(pdf_name: str) -> Iterator[fitz.Pixmap]:
    """Transform from pdf file to page images, return pixmap images one-by-one (generator)"""
    doc = fitz.open(pdf_name)
    for page in doc:
        yield page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), colorspace=fitz.csGRAY)


def skip_line(thr_img: np.ndarray, ind_str_pix: int, is_string: bool = True, left_border: int = None,
              right_border: int = None, end_height: int = None) -> int:
    """Return the vertical coordinate of changing text and empty (text->empty, is_string = False another)"""
    if end_height is None:
        end_height = thr_img.shape[0]
    while is_string == (sum(thr_img[ind_str_pix, left_border:right_border]) != 0) and end_height > ind_str_pix:
        ind_str_pix += 1
    return ind_str_pix


def skip_string(thr_img: np.ndarray, ind_str_pix: int, left_border: int = None, right_border: int = None,
                end_height: int = None) -> Tuple[int, int]:
    """Go from vertical beginning of string to next string, return coord of next line and dist between lines"""
    end_of_first_line = skip_line(thr_img, ind_str_pix, True, left_border, right_border, end_height)
    start_of_second_line = skip_line(thr_img, end_of_first_line, False, left_border, right_border, end_height)
    dist_between_lines = start_of_second_line - end_of_first_line
    return start_of_second_line, dist_between_lines


def find_heading(cv_img: np.ndarray, first_str: str, start_height: int = 0, end_height: int = None,
                 count_of_str: int = None, left_border: int = None, right_border: int = None) -> np.ndarray:
    """Find head with first string in input image, return found image with head"""
    if end_height is None:
        end_height = cv_img.shape[0]
    thr_img = cv.threshold(cv_img, 200, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)[1]

    # go to the first string
    ind_str_pix = skip_line(thr_img, start_height, False, left_border, right_border, end_height)
    # go to the second string
    ind_str_pix, dist_between_lines = skip_string(thr_img, ind_str_pix, left_border, right_border, end_height)
    # check name
    if ind_str_pix >= end_height or recog_text_full(cv_img[start_height:ind_str_pix - 1]).strip() != first_str:
        return np.array([])

    cur_count = 1
    small_dist = dist_between_lines
    if count_of_str is None:
        while dist_between_lines < 2 * small_dist:
            ind_str_pix, dist_between_lines = skip_string(thr_img, ind_str_pix, left_border, right_border)
    else:
        while cur_count < count_of_str:
            ind_str_pix, dist_between_lines = skip_string(thr_img, ind_str_pix, left_border, right_border)
            cur_count += 1
    return cv_img[start_height:ind_str_pix - 1, left_border:right_border]


def print_res(r: dict, out_file: str = None) -> None:
    """Print dictionary into file or stdout with formatting"""
    if output_file is not None:
        f = open(out_file, "w")
    else:
        f = sys.stdout

    print(len(r["forms"]), file=f)
    for form in r["forms"]:
        print(str(form["okpo"]) + ";" + str(form["name"]) + ";" + str(form["inn"]) + ";" + str(
            form["time"]) + ";", file=f)
    if "Au" in r.keys():
        print("au", file=f)
        print(r["Au"].replace("\n", " ") + ";", file=f)
    if "po" in r.keys():
        print("po", file=f)
        print(r["po"].replace("\n", " ") + ";", file=f)


def print_res_test(r: dict, in_file: str, out_file: str = None) -> None:
    """Print dictionary into file or stdout with formatting"""
    if out_file is not None:
        f = open(out_file, "w")
    else:
        f = sys.stdout

    res_str = in_file + ";"

    if "Au" in r.keys():
        res_str += r["Au"].replace("\n", " ") + ";"
    else:
        res_str += ";"
    if "po" in r.keys():
        res_str += r["po"].replace("\n", " ") + ";"
    else:
        res_str += ";"

    for form in r["forms"]:
        res_str += str(form["okpo"]) + ";" + str(form["name"]) + ";" + str(form["inn"]) + ";" + str(
            form["time"]) + ";"
    print(res_str, file=f)


def find_small_table(thr_inv_img: np.ndarray) -> List[int]:
    """Find in contours in input binary image small table, return [x_min, x_max, y_min, y_max]"""
    contours0 = cv.findContours(thr_inv_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    for cont in contours0:
        if cv.contourArea(cont) < 50000:
            continue
        rect = cv.minAreaRect(cont)  # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат\
        area = cv.contourArea(box)  # вычисление площади
        if 50000 < area < 500000 and math.fabs(area - cv.contourArea(cont)) < 40000:
            return [min((coord[0] for coord in box)), max((coord[0] for coord in box)),
                    min((coord[1] for coord in box)), max((coord[1] for coord in box))]
    return []


def points_of_table(table_img: np.ndarray, templ_1: np.ndarray, templ_2: np.ndarray) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    """In input image find vertical and gorizontal lines (templs) and return it's coordinates"""
    gor_points = sorted(find_templ(table_img, templ_1), key=lambda coord: coord[1])
    vert_points = sorted(find_templ(table_img, templ_2), key=lambda coord: coord[0])
    for i_ in range(len(vert_points) - 1, 0, -1):
        if vert_points[i_][0] - vert_points[i_ - 1][0] < 10:
            del vert_points[i_]

    if len(vert_points) < 2 or len(vert_points) > 3:
        return None, None

    for i_ in range(len(gor_points) - 1, 0, -1):
        if not (vert_points[0][0] <= gor_points[i_][0] <= vert_points[-1][0]):
            del gor_points[i_]

    for i_ in range(len(gor_points) - 1, 0, -1):
        if gor_points[i_][1] - gor_points[i_ - 1][1] < 10:
            del gor_points[i_]

    return [(vert_points[0][0], coord[1]) for coord in gor_points][1:-1], \
           [(vert_points[-1][0], coord[1]) for coord in gor_points][1:-1]


def get_gor_line_template(line_len: int) -> np.ndarray:
    """Return the image with gorizontal line"""
    templ = np.zeros((3, line_len), dtype=np.uint8)
    templ[0] += 255
    templ[2] += 255
    return templ


def get_vert_line_template(line_len: int) -> np.ndarray:
    """Return the image with vertical line"""
    return get_gor_line_template(line_len).transpose()


def get_corner_template(type_corner: int, width: int, height: int) -> np.ndarray:
    """Return image with corner: 0 - left up, 1 - right up, 2 - left down, 3 - right down"""
    templ = np.zeros((height, width), dtype=np.uint8) + 255
    if type_corner == 0 or type_corner == 1:
        templ[0] *= 0
    else:
        templ[height - 1] *= 0

    if type_corner == 0 or type_corner == 2:
        templ[:, 0] *= 0
    else:
        templ[:, width - 1] *= 0
    return templ


def first_cut_and_rotate(in_img: np.ndarray) -> Optional[np.ndarray]:
    """Return aligned image with cutting borders"""
    in_img = in_img[25:-25, 25:-25]
    thr_img = cv.threshold(cv.medianBlur(in_img, 3), 220, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    k = 0.5
    lines = cv.HoughLines(thr_img, 1, np.pi / 360.0, round((thr_img.shape[0]) * k))
    if lines is not None:
        angls = [lines[0][0][1], lines[0][0][1] - math.pi / 2,
                 lines[0][0][1] - math.pi, lines[0][0][1] - 3 * math.pi / 2]
        angle_ind = np.argmin(np.fabs(np.array(angls)))
        angle = math.degrees(angls[angle_ind]) if angls[angle_ind] >= 0 else math.degrees(
            2 * math.pi + angls[angle_ind])
    else:
        return None

    return rotation(in_img, angle)


def find_corners(thr_img: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return finding corners in input image"""
    w = 100
    h = 50
    if thr_img.sum() / 255 < 8400000:
        return [], [], [], []
    lu_c = [((p[0] - w // 2, p[1] - h // 2), 0) for p in find_templ(thr_img, get_corner_template(0, w, h))]
    if len(lu_c) == 0:
        return [], [], [], []
    ru_c = [((p[0] + w // 2, p[1] - h // 2), 1) for p in find_templ(thr_img, get_corner_template(1, w, h))]
    if len(ru_c) == 0:
        return [], [], [], []
    ld_c = [((p[0] - w // 2, p[1] + h // 2), 2) for p in find_templ(thr_img, get_corner_template(2, w, h))]
    if len(ld_c) == 0:
        return [], [], [], []
    rd_c = [((p[0] + w // 2, p[1] + h // 2), 3) for p in find_templ(thr_img, get_corner_template(3, w, h))]
    if len(rd_c) == 0:
        return [], [], [], []

    corners = lu_c + ru_c + ld_c + rd_c
    i_ = 0
    while i_ < len(corners):
        del_ind = [j for j in range(i_ + 1, len(corners)) if
                   (corners[i_][0][0] - corners[j][0][0]) ** 2 + (corners[i_][0][1] - corners[j][0][1]) ** 2 < 100 and
                   corners[i_][1] != corners[j][1]]
        if len(del_ind) > 0:
            del_ind.reverse()
            del_ind.append(i_)
            for j in del_ind:
                del corners[j]
        else:
            i_ += 1

    return ([p[0] for p in corners if p[1] == 0], [p[0] for p in corners if p[1] == 1],
            [p[0] for p in corners if p[1] == 2], [p[0] for p in corners if p[1] == 3])


def draw_rect(in_img: np.ndarray, coord: tuple) -> np.ndarray:
    return cv.rectangle(in_img, (coord[0], coord[1]), (coord[0] + 20, coord[1] + 20), 0, 2, 8, 0)


def sqr_dist_between_points(pt1: tuple, pt2: tuple) -> float:
    return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2


def find_table_in_corners(lu_c: List[Tuple[int, int]],
                          ru_c: List[Tuple[int, int]],
                          ld_c: List[Tuple[int, int]],
                          rd_c: List[Tuple[int, int]]) -> List[int]:
    """Find in corners small table, return [x_min, x_max, y_min, y_max]"""
    for cur_lu_c in lu_c:
        for cur_ru_c in ru_c:
            if not (math.fabs(cur_ru_c[1] - cur_lu_c[1]) < 20 and 100 < cur_ru_c[0] - cur_lu_c[0] < 800):
                continue
            for cur_ld_c in ld_c:
                if not (math.fabs(cur_ld_c[0] - cur_lu_c[0]) < 20 and 100 < cur_ld_c[1] - cur_lu_c[1] < 800):
                    continue
                for cur_rd_c in rd_c:
                    if math.fabs(cur_rd_c[1] - cur_ld_c[1]) < 20 and 100 < cur_rd_c[0] - cur_ld_c[
                        0] < 800 and math.fabs(cur_rd_c[0] - cur_ru_c[0]) < 20 and 100 < cur_rd_c[1] - cur_ru_c[
                        1] < 800:
                        return [min(cur_lu_c[0], cur_ld_c[0]), max(cur_ru_c[0], cur_rd_c[0]),
                                min(cur_lu_c[1], cur_ru_c[1]), max(cur_ld_c[1], cur_rd_c[1])]
    return []


def find_head_and_table(in_img: np.ndarray) -> dict:
    """In input img find small table and head"""
    res = {}
    cv_img = first_cut_and_rotate(in_img)
    if cv_img is None:
        return res
    thr_img = cv.threshold(cv.medianBlur(cv_img, 3), 220, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    edges = find_small_table(255 - thr_img)
    if len(edges) == 0:
        edges = find_table_in_corners(*find_corners(thr_img))
    if len(edges) == 0:
        return res
    angles = [0, 180, 270, 90]
    res_angle = angles[quad_of_table([cv_img.shape[1], cv_img.shape[0]], edges)]
    if res_angle == 90:
        edges = [edges[2], edges[3], cv_img.shape[1] - edges[1], cv_img.shape[1] - edges[0]]
    elif res_angle == 180:
        edges = [cv_img.shape[1] - edges[1], cv_img.shape[1] - edges[0], cv_img.shape[0] - edges[3],
                 cv_img.shape[0] - edges[2]]
    elif res_angle == 270:
        edges = [cv_img.shape[0] - edges[3], cv_img.shape[0] - edges[2], edges[0], edges[1]]
    first_rot_img = rotation(cv_img, res_angle)

    res_img = first_rot_img[20:edges[3] + 20, 20:edges[1] + 20]
    table_img = res_img[edges[2] - 30:edges[3] + 10, edges[0] - 30:edges[1] + 10]
    thr_table = cv.threshold(cv.medianBlur(table_img, 3), 220, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    left_points, right_points = points_of_table(thr_table, get_gor_line_template(100), get_vert_line_template(100))

    if left_points is None or len(left_points) < 4 or len(left_points) > 10:
        return res

    okud_title = None
    inn_title = None
    end_titles = edges[0] - 30 + left_points[0][0] - 2
    thr_res = cv.threshold(cv.medianBlur(res_img, 3), 220, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    for i_ in range(len(left_points) - 1):
        pos = end_titles
        while thr_res[edges[2] - 30 + left_points[i_][1]:edges[2] - 30 + left_points[i_ + 1][1],
              pos].sum() < 5 and pos > end_titles - 500:
            pos -= 5
        if pos <= end_titles - 500:
            continue
        res_str = recog_text_full(cv.copyMakeBorder(
            res_img[edges[2] - 30 + left_points[i_][1]:edges[2] - 30 + left_points[i_ + 1][1], pos - 200:pos + 5], 20,
            20, 20, 20,
            cv.BORDER_CONSTANT, value=255)).upper()
        if res_str.find("ОКУД") != -1:
            okud_title = i_
        elif res_str.find("ИНН") != -1:
            inn_title = i_

    # cut okpo
    if okud_title is not None:
        x_1_1 = (left_points[okud_title][0] + left_points[okud_title + 1][0]) // 2 + edges[0] - 30
        x_1_2 = (right_points[okud_title][0] + right_points[okud_title + 1][0]) // 2 + edges[0] - 30
        y_1_1 = (left_points[okud_title][1] + right_points[okud_title][1]) // 2 + edges[2] - 30
        y_1_2 = (left_points[okud_title + 1][1] + right_points[okud_title + 1][1]) // 2 + edges[2] - 30
        res['okpo'] = cv.copyMakeBorder(res_img[y_1_1 + 7:y_1_2, x_1_1 + 7:x_1_2], 20, 20, 20, 20,
                                        cv.BORDER_CONSTANT,
                                        value=255)
    else:
        res['okpo'] = None

    # cut
    if inn_title is not None:
        x_1_1 = (left_points[inn_title][0] + left_points[inn_title][0]) // 2 + edges[0] - 30
        x_1_2 = (right_points[inn_title][0] + right_points[inn_title + 1][0]) // 2 + edges[0] - 30
        y_1_1 = (left_points[inn_title][1] + right_points[inn_title][1]) // 2 + edges[2] - 30
        y_1_2 = (left_points[inn_title + 1][1] + right_points[inn_title + 1][1]) // 2 + edges[2] - 30
        res['inn'] = cv.copyMakeBorder(res_img[y_1_1 + 5:y_1_2 - 2, x_1_1 + 7:x_1_2], 20, 20, 20, 20,
                                       cv.BORDER_CONSTANT,
                                       value=255)
    else:
        res['inn'] = None

    res['head'] = res_img[50: edges[2] - 30 + left_points[2][1] - 10, 200: edges[0] - 30 + left_points[2][0] - 100]
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    file_name = sys.argv[2]
    output_file = None if len(sys.argv) < 4 else sys.argv[3]
    # for file_name in [os.path.join(path, name) for name in os.listdir(path) if
    #                   os.path.splitext(name)[1] == ".pdf" or os.path.splitext(name)[1] == ".PDF"]:
    #     res_str = os.path.split(file_name)[1] + ";"

    res = {"forms": []}
    numb_of_page = 0
    for page in load_file_using_fitz(file_name):
        numb_of_page += 1
        if numb_of_page > 20:
            break
        img = np.reshape(np.frombuffer(page.samples, dtype=np.uint8), [page.height, page.width])
        table = find_head_and_table(img)
        if bool(table):
            res["forms"].append(table)
            continue
        if mode != 2:
            head = find_heading(img, "Аудиторское заключение", 250, 400, None, 50, -50)
            if head.size != 0:
                res["Au"] = head
                continue
            head = find_heading(img, "АУДИТОРСКОЕ ЗАКЛЮЧЕНИЕ", 1200, 2200, None, 400, -400)
            if head.size != 0:
                res["Au"] = head
                continue
            head = find_heading(img, "ПОЯСНЕНИЯ", 300, 600, None, 300, -200)
            if head.size != 0:
                res["po"] = head
                break

    for i in range(len(res["forms"])):
        res["forms"][i]['okpo'] = recog_numb_full(res["forms"][i]['okpo']) if res["forms"][i][
                                                                                  'okpo'] is not None else None
        res["forms"][i]['inn'] = recog_numb_full(res["forms"][i]['inn']) if res["forms"][i][
                                                                                'inn'] is not None else None
        res["forms"][i]['head'] = recog_text_full(res["forms"][i]['head']).replace("\n\n", "\n").split("\n")[:2]
        res["forms"][i]['name'] = res["forms"][i]['head'][0].replace(";", "").replace("\"", "")
        if len(res["forms"][i]['head']) > 1:
            res["forms"][i]['time'] = res["forms"][i]['head'][1].replace(";", "").replace("\"", "")
        else:
            res["forms"][i]['time'] = res["forms"][i]['head'][0].replace(";", "").replace("\"", "")
        del res["forms"][i]['head']

    if "Au" in res.keys():
        res["Au"] = recog_text_full(res["Au"]).replace("\n\n", "\n").replace(";", "").replace("\"", "")
    if "po" in res.keys():
        res["po"] = recog_text_full(res["po"]).replace("\n\n", "\n").replace(";", "").replace("\"", "")

    print_res(res, output_file)

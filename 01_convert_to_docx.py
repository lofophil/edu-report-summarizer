# pip install comtypes tqdm python-docx pdf2docx
import os, shutil, tempfile, subprocess, csv, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

DOC_EXTS = {".doc", ".docx"}
PDF_EXTS = {".pdf"}

def word_save_as_docx(src_path: Path, dst_path: Path) -> None:
    import comtypes.client
    word = comtypes.client.CreateObject('Word.Application')
    # 降低弹窗/交互
    word.Visible = False
    word.DisplayAlerts = 0  # wdAlertsNone
    try:
        # 优先尝试“修复打开”
        doc = word.Documents.Open(
            FileName=str(src_path),
            ReadOnly=True,
            ConfirmConversions=False,
            AddToRecentFiles=False,
            OpenAndRepair=True,
            Visible=False
        )
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            wdFormatXMLDocument = 16
            doc.SaveAs(str(dst_path), FileFormat=wdFormatXMLDocument)
        finally:
            doc.Close(False)
    finally:
        word.Quit()

def try_soffice_docx(src_path: Path, dst_path: Path) -> bool:
    # 需要本机已安装 LibreOffice，并在 PATH 中有 soffice
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "soffice", "--headless", "--convert-to", "docx", "--outdir",
            str(dst_path.parent), str(src_path)
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        return r.returncode == 0 and dst_path.exists()
    except Exception:
        return False

def try_pdf2docx(src_path: Path, dst_path: Path) -> bool:
    try:
        from pdf2docx import Converter
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv = Converter(str(src_path))
        cv.convert(str(dst_path), start=0, end=None)
        cv.close()
        return dst_path.exists()
    except Exception:
        return False

def convert_one(args):
    src_path, dst_path = args
    suffix = src_path.suffix.lower()

    # 跳过已最新
    if dst_path.exists() and dst_path.stat().st_mtime >= src_path.stat().st_mtime:
        return True, str(src_path), "skipped(up-to-date)"

    # 1) 先尝试原路径打开
    try:
        word_save_as_docx(src_path, dst_path)
        return True, str(src_path), "word_ok"
    except Exception as e1:
        # 2) 临时 ASCII 文件名再试（规避某些安全策略/奇异字符）
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / ("tmp" + suffix)
                shutil.copy2(src_path, tmp)
                word_save_as_docx(tmp, dst_path)
                return True, str(src_path), "word_ok_tmpname"
        except Exception as e2:
            # 3) LibreOffice 兜底
            try:
                ok = try_soffice_docx(src_path, dst_path)
                if ok:
                    return True, str(src_path), "soffice_ok"
            except Exception:
                pass

            # 4) 仅 PDF 再试 pdf2docx
            if suffix in PDF_EXTS:
                if try_pdf2docx(src_path, dst_path):
                    return True, str(src_path), "pdf2docx_ok"

            # 全部失败
            return False, f"{src_path}", f"fail: {repr(e1)} | tmp_retry: {repr(e2)}"

def batch_convert_parallel(root_dir, out_dir, workers=16, log_path=None):
    root = Path(root_dir)
    out  = Path(out_dir)
    files = [p for p in root.rglob("*") if p.suffix.lower() in DOC_EXTS.union(PDF_EXTS)]
    tasks = []
    for p in files:
        rel = p.relative_to(root)
        dst = out / rel.with_suffix(".docx")
        tasks.append((p, dst))

    ok, fail = 0, 0
    fails = []
    stats = {"word_ok":0, "word_ok_tmpname":0, "soffice_ok":0, "pdf2docx_ok":0, "skipped(up-to-date)":0}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(convert_one, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Converting x{workers}"):
            succ, msg, tag = f.result()
            if succ:
                ok += 1
                stats[tag] = stats.get(tag, 0) + 1
            else:
                fail += 1
                fails.append((msg, tag))
                print("[FAIL]", msg, "=>", tag)

    print(f"\nDone. success={ok}, fail={fail}")
    print("stats:", stats)

    if log_path:
        with open(log_path, "w", newline="", encoding="utf-8-sig") as w:
            cw = csv.writer(w)
            cw.writerow(["src_path", "error"])
            cw.writerows(fails)

if __name__ == "__main__":
    batch_convert_parallel(
        r"D:\语义分析模型训练\PythonProject\研究报告",
        r"D:\语义分析模型训练\PythonProject\data",
        workers=16,  # 先用 16；如稳定再提升到 24/32
        log_path=r"D:\语义分析模型训练\PythonProject\convert_failures.csv"
    )

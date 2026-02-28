import sys
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from data import Storage, WorkbookService
from analysis import Analyzer
from report import ReportGenerator

# Use relative path for data directory
storage = Storage(str(script_dir.parent / "data"))
ws = WorkbookService(storage)
analyzer = Analyzer(storage)
report_gen = ReportGenerator(storage, ws, analyzer)

html_file = report_gen.export_html("wb_dc3f94ab4aa4")
print(f"Generated: {html_file}")

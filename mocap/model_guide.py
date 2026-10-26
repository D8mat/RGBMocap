"""Compact model comparison using the same recorded seven-second clip."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices, QTextOption
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QPushButton, QTextBrowser, QVBoxLayout
from .motion_tools import MOTION_TOOLS

# exmr-eval-2026-10-22/summary-body.json, clip_01 (210 frames at 30 fps).
# Historical processing-only runs; FlowHMR predates Fast SAM.
BENCHMARKS = {
    'gvhmr': ('2m 01s', '5.08', '12.47'),
    'flowhmr': ('2m 49s', '14.89', '22.19'),
    'exmr': ('1m 08s', '11.32', '19.31'),
}
PROJECT_URLS = {
    'gvhmr': 'https://github.com/zju3dv/GVHMR',
    'flowhmr': 'https://github.com/flowhmr/flowhmr',
    'exmr': 'https://github.com/twehrbein/ExMR',
}
DESCRIPTIONS = {
    'gvhmr': 'Lowest overall motion quality in our tests, but sometimes better for unusual poses such as backflips. '
             'No native finger tracking. Optional HandFlow adds finger motion; FootMR refines ankles, and HTD refines body motion. These add processing time.',
    'flowhmr': 'Slow, but the best overall motion quality in our tests. Tracks body and fingers together without a separate finger add-on.',
    'exmr': 'Generally fast, with good motion quality in our tests. A practical choice for a first attempt. Tracks body and fingers together without a separate finger add-on.',
}


def guide_html(selected):
    descriptions = ''.join(
        f'<p><b>{tool.label}</b> &nbsp; <a href="{PROJECT_URLS[key]}">GitHub project</a><br>'
        f'{DESCRIPTIONS[key]}</p>' for key, tool in MOTION_TOOLS.items())
    rows = ''.join(
        ('<tr bgcolor="#303f50">' if key == selected else '<tr>') +
        f'<td><b>{MOTION_TOOLS[key].label}</b></td>' +
        ''.join(f'<td>{value}</td>' for value in values) + '</tr>'
        for key, values in BENCHMARKS.items())
    return """<html><head><style>
body { font-size:12px; } h3 { margin-top:15px; } a { color:#8ed2ff; }
th { text-align:left; background-color:#333333; } td, th { padding:7px; }
</style></head><body>
<h3>Choosing a model</h3>
<p>These are subjective observations from our test videos, not a universal ranking or accuracy benchmark.
Try another model if a particular movement looks wrong.</p>""" + descriptions + """
<h3>Recorded speed test: HIGHEST BACKFLIP · 7 seconds</h3>
<table width="100%" border="0" cellspacing="1" cellpadding="7">
<tr><th>Model</th><th>Processing time</th><th>Peak VRAM (GiB)</th><th>Peak RAM (GiB)</th></tr>""" + rows + """</table>
<p>Test PC: RTX 5090 32 GB · Ryzen 9 9950X · 96 GB RAM.</p>
<p>GVHMR: HTD + FootMR + HandFlow on. FlowHMR: original SAM; today's Fast SAM can
be faster. Recorded processing only, excluding tracking, app correction and export.
Memory shows measured worker peaks, not minimum requirements. Results vary by video and PC.</p>
<p>All three support Motion Correction for foot and hand contacts. Contact correction reduces sliding;
it is separate from tracking fingers or refining ankle angles.</p></body></html>"""


class ModelGuideDialog(QDialog):
    def __init__(self, parent, selected):
        super().__init__(parent)
        self.setWindowTitle('Motion Model Guide')
        self.setMinimumSize(360, 260)
        area = self.screen().availableGeometry()
        self.resize(min(740, max(360, area.width()-80)),
                    min(540, max(260, area.height()-100)))
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setAccessibleName('Motion model descriptions and short-clip benchmarks')
        self.browser.setStyleSheet('QTextBrowser { font-family:"' + QApplication.font().family() + '"; font-size:12px; padding:6px; }')
        self.browser.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.browser.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(QDesktopServices.openUrl)
        self.browser.setHtml(guide_html(selected))
        layout.addWidget(self.browser)
        actions = QHBoxLayout(); actions.addStretch()
        close = QPushButton('Close'); close.clicked.connect(self.accept)
        close.setToolTip('Close the model guide.')
        actions.addWidget(close); layout.addLayout(actions)

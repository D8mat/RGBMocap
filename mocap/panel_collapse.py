"""Collapsible workspace panels; retain child visibility and splitter sizes."""
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLayout

class PanelCollapse(QObject):
    def __init__(self, panel, splitter, title, header=None):
        super().__init__(panel)
        self.panel,self.splitter,self.title=panel,splitter,title
        self.horizontal=splitter.orientation()==Qt.Horizontal
        self.minimum=panel.minimumSize();self.maximum=panel.maximumSize()
        self.saved_sizes=None
        layout=panel.layout()
        layout.setSizeConstraint(QLayout.SetNoConstraint)
        if self.horizontal:layout.setAlignment(Qt.AlignTop)
        content=QWidget(panel);box=QVBoxLayout(content)
        box.setContentsMargins(0,0,0,0);box.setSpacing(layout.spacing())
        first=1 if header is not None else 0
        while layout.count()>first:
            item=layout.takeAt(first)
            if item.widget():box.addWidget(item.widget(),item.stretch() if hasattr(item,'stretch') else 0)
            elif item.layout():box.addLayout(item.layout())
            else:box.addItem(item)
        self.content=content
        self.button=QPushButton();self.button.setCheckable(True);self.button.setChecked(True)
        self.button.setAccessibleName('Collapse '+title)
        self.button.setStyleSheet('QPushButton { text-align: left; padding: 4px; }')
        if header is not None:
            old=header.takeAt(0)
            if old and old.widget():old.widget().deleteLater()
            header.insertWidget(0,self.button)
        else:layout.insertWidget(0,self.button)
        layout.addWidget(content,1)
        self.button.toggled.connect(self.toggle)
        self.refresh_label(True)
    def refresh_label(self,expanded):
        self.button.setText(('▼ '+self.title) if expanded else ('▶' if self.horizontal else '▶ '+self.title))
        self.button.setToolTip(('Collapse ' if expanded else 'Expand ')+self.title)
        self.button.setAccessibleName(('Collapse ' if expanded else 'Expand ')+self.title)
    def toggle(self,expanded):
        if not expanded:self.saved_sizes=self.splitter.sizes()
        self.content.setVisible(expanded)
        self.refresh_label(expanded)
        if expanded:
            self.panel.setMinimumSize(self.minimum);self.panel.setMaximumSize(self.maximum)
        else:
            margins=self.panel.layout().contentsMargins()
            if self.horizontal:self.panel.setFixedWidth(36+margins.left()+margins.right())
            else:
                header=self.panel.layout().itemAt(0).widget()
                self.panel.setFixedHeight(header.sizeHint().height()+margins.top()+margins.bottom())
        self.panel.layout().invalidate()
        self.panel.updateGeometry()
        self.reflow()
        QTimer.singleShot(0,self.reflow)
    def reflow(self):
        expanded=self.button.isChecked()
        sizes=self.splitter.sizes();index=self.splitter.indexOf(self.panel)
        if index<0:return
        extent=self.splitter.width() if self.horizontal else self.splitter.height()
        total=max(0,extent-self.splitter.handleWidth()*(len(sizes)-1))
        if expanded and self.saved_sizes:
            target=self.saved_sizes[index]
        else:target=self.panel.width() if self.horizontal else self.panel.height()
        others=[i for i in range(len(sizes)) if i!=index]
        if not others:return
        available=max(0,total-target);weight=sum(sizes[i] for i in others)
        for i in others:sizes[i]=round(available*sizes[i]/weight) if weight else available//len(others)
        sizes[index]=target
        self.splitter.setSizes(sizes)
        parent=self.splitter.parentWidget()
        while parent is not None:
            if parent.layout():parent.layout().invalidate();parent.layout().activate()
            parent.updateGeometry();parent=parent.parentWidget()

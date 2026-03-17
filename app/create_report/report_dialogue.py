"""
Dialog for selecting which dataset keys to include in a hole-level PDF booklet.

Allows the user to select from datasets that are present in ALL boxes within
the hole, ensuring consistent output across the entire booklet.
"""
import logging
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QDialogButtonBox,
    QMessageBox,
)

logger = logging.getLogger(__name__)


class PDFExportDialog(QDialog):
    """
    Dialog for selecting dataset keys to include in PDF booklet.
    
    Presents a list of dataset keys that are available in ALL boxes,
    allows multi-selection, and returns the selected keys on accept.
    """
    
    def __init__(self, cxt, parent=None):
        """
        Parameters
        ----------
        cxt : CurrentContext
            Current application context containing the loaded hole
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.cxt = cxt
        self.setWindowTitle("Export Hole PDF Booklet")
        self.selected_keys = []
        
        # Get available keys from the hole
        available_keys = self._get_keys_in_all_boxes()
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Select dataset keys to include in the PDF booklet.\n"
            "Each selected key will generate one page per box.\n"
            "Only datasets present in ALL boxes are shown."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # List widget for multi-selection
        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        
        # Add items
        for key in available_keys:
            item = QListWidgetItem(key)
            self.list_widget.addItem(item)
        
        layout.addWidget(self.list_widget)
        
        # Selection buttons
        btn_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        btn_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        btn_layout.addWidget(deselect_all_btn)
        
        layout.addLayout(btn_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)
    
    def _get_keys_in_all_boxes(self) -> list[str]:
        """Find dataset keys that are present in ALL boxes."""
        valid_state, msg = self.cxt.requires(self.cxt.HOLE)
        if not valid_state:
            return []
        
        hole = self.cxt.ho
        if not hole.boxes:
            return []
        
        # Start with keys from first box
        first_box = list(hole.boxes.values())[0]
        all_keys = set(first_box.datasets.keys()) | set(first_box.temp_datasets.keys())
        
        # Intersect with keys from all other boxes
        for po in hole:
            box_keys = set(po.datasets.keys()) | set(po.temp_datasets.keys())
            all_keys &= box_keys
        
        # Filter out non-visualizable keys
        excluded_keys = {'metadata', 'bands', 'display', 'stats'}
        visualizable_keys = [
            k for k in all_keys 
            if k not in excluded_keys 
            and not k.endswith("LEGEND") 
            and not k.lower().startswith("prof")
        ]
        
        return sorted(visualizable_keys)
    
    def select_all(self):
        """Select all items in the list."""
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setSelected(True)
    
    def deselect_all(self):
        """Deselect all items in the list."""
        self.list_widget.clearSelection()
    
    def accept(self):
        """Override accept to validate selection, get output path, and generate PDF."""
        selected_items = self.list_widget.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select at least one dataset key to include in the PDF."
            )
            return
        
        self.selected_keys = [item.text() for item in selected_items]
        logger.info(f"User selected {len(self.selected_keys)} keys for PDF export: {self.selected_keys}")
        
        # Ask where to save the PDF
               
        hole = self.cxt.ho
        default_filename = f"{hole.hole_id}_booklet.pdf"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF Booklet",
            str(hole.root_dir / default_filename),
            "PDF Files (*.pdf)"
        )
        
        if not output_path:
            logger.info("PDF export cancelled - no output path selected")
            return
        
        # Generate the PDF
        try:
            from .pdf_booklet import create_hole_pdf_booklet
            from ..ui.util_windows import busy_cursor
            
            with busy_cursor('Generating PDF booklet...', self):
                pdf_path = create_hole_pdf_booklet(
                    hole=hole,
                    selected_keys=self.selected_keys,
                    output_path=output_path
                )
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"PDF booklet created successfully:\n{pdf_path}"
            )
            logger.info(f"PDF booklet exported to {pdf_path}")
            super().accept()
            
        except Exception as e:
            logger.error(f"Failed to create PDF booklet: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Export Failed", str(e))
            
    
    def get_selected_keys(self) -> list[str]:
        """
        Get the list of selected keys.
        
        Returns
        -------
        list[str]
            Selected dataset keys
        """
        return self.selected_keys
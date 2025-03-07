"""
Whisper GUI Application - Main entry point
"""

import sys
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox

from whisper_gui.ui.main_window import WhisperGUI

def main():
    """
    Main application entry point
    """
    try:
        app = QApplication(sys.argv)
        window = WhisperGUI()
        window.show()
        sys.exit(app.exec())  # PyQt6에서는 app.exec_() 대신 app.exec()를 사용
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {str(e)}")
        traceback.print_exc()
        
        # 메시지 박스로 오류 표시
        if QApplication.instance():
            QMessageBox.critical(None, "심각한 오류", f"프로그램 실행 중 오류가 발생했습니다:\n{str(e)}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
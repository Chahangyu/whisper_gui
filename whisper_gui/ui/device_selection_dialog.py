"""
장치 선택 다이얼로그 - GPU 또는 CPU 모드 선택
"""

import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QRadioButton, QButtonGroup,
                             QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont

class DeviceSelectionDialog(QDialog):
    """사용자가 GPU 또는 CPU 모드를 선택할 수 있는 다이얼로그"""
    
    # 장치 선택 완료 시 신호 전송
    device_selected = pyqtSignal(bool)  # True: GPU, False: CPU
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Whisper - 장치 선택")
        self.setFixedSize(500, 300)
        self.setModal(True)
        
        # 필요한 DLL 파일 확인
        self.check_required_dlls()
        
        # UI 초기화
        self.init_ui()
        
    def check_required_dlls(self):
        """필요한 DLL 파일의 존재 여부 확인"""
        self.current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Vulkan DLL 확인
        self.vulkan_available = os.path.exists(os.path.join(self.current_dir, "ggml-vulkan.dll"))
        
        # 기본 DLL 확인
        self.required_dlls = ["whisper.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"]
        self.missing_dlls = []
        
        for dll in self.required_dlls:
            if not os.path.exists(os.path.join(self.current_dir, dll)):
                self.missing_dlls.append(dll)
    
    def init_ui(self):
        """UI 구성"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # 타이틀 레이블
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        
        title_label = QLabel("Whisper 음성 인식 장치 선택")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 설명 텍스트
        desc_label = QLabel("음성 인식에 사용할 하드웨어를 선택하세요. GPU 가속을 사용하면 더 빠른 속도로 처리할 수 있습니다.")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)
        
        # 장치 선택 그룹박스
        device_group = QGroupBox("처리 장치")
        device_layout = QVBoxLayout()
        
        # 라디오 버튼 그룹
        self.btn_group = QButtonGroup()
        
        # GPU 선택 버튼
        self.gpu_radio = QRadioButton("GPU 가속 (Vulkan)")
        self.gpu_radio.setEnabled(self.vulkan_available)
        if self.vulkan_available:
            self.gpu_radio.setChecked(True)
            self.gpu_radio.setToolTip("Vulkan을 사용하여 GPU에서 처리합니다 (권장)")
        else:
            self.gpu_radio.setToolTip("ggml-vulkan.dll이 없어 GPU 가속을 사용할 수 없습니다")
        self.btn_group.addButton(self.gpu_radio)
        device_layout.addWidget(self.gpu_radio)
        
        # CPU 선택 버튼
        self.cpu_radio = QRadioButton("CPU 모드")
        self.cpu_radio.setToolTip("CPU를 사용하여 처리합니다 (느릴 수 있음)")
        if not self.vulkan_available:
            self.cpu_radio.setChecked(True)
        self.btn_group.addButton(self.cpu_radio)
        device_layout.addWidget(self.cpu_radio)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # 상태 메시지
        self.status_label = QLabel()
        if self.missing_dlls:
            self.status_label.setText(f"경고: 일부 필수 DLL 파일이 없습니다: {', '.join(self.missing_dlls)}")
            self.status_label.setStyleSheet("color: red;")
        elif not self.vulkan_available:
            self.status_label.setText("GPU 가속을 사용할 수 없습니다. CPU 모드로 실행됩니다.")
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setText("모든 필수 DLL 파일이 로드되었습니다. GPU 및 CPU 모드를 사용할 수 있습니다.")
            self.status_label.setStyleSheet("color: green;")
        layout.addWidget(self.status_label)
        
        # 버튼
        btn_layout = QHBoxLayout()
        
        exit_btn = QPushButton("종료")
        exit_btn.clicked.connect(self.reject)
        
        next_btn = QPushButton("다음")
        next_btn.clicked.connect(self.on_next_clicked)
        next_btn.setDefault(True)
        
        btn_layout.addWidget(exit_btn)
        btn_layout.addWidget(next_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def on_next_clicked(self):
        """다음 버튼 클릭 처리"""
        # 필수 DLL이 없는 경우 경고
        if self.missing_dlls:
            reply = QMessageBox.warning(
                self, 
                "필수 파일 없음", 
                f"일부 필수 파일이 없습니다: {', '.join(self.missing_dlls)}\n계속 진행하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # GPU 또는 CPU 선택 값 전달
        use_gpu = self.gpu_radio.isChecked() and self.vulkan_available
        self.device_selected.emit(use_gpu)
        self.accept()
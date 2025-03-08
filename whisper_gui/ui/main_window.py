"""
WhisperGUI class - Main application window and user interface
"""

import os
import sys
from PyQt6.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QLabel, QComboBox, QTextEdit, 
                             QFileDialog, QProgressBar, QMessageBox, QApplication)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..core.whisper_dll import WhisperDLL
from ..core.recording_thread import RecordingThread
from ..core.transcription_thread import TranscriptionThread
from ..core.utils import find_dll_file, check_required_dlls
from ..core.model_downloader import ModelDownloader

class WhisperGUI(QMainWindow):
    """Whisper 음성 인식 GUI 메인 클래스"""
    def __init__(self):
        super().__init__()
        
        # 윈도우 설정
        self.setWindowTitle("Whisper 다국어 음성 인식 프로그램 (Vulkan 지원)")
        self.setGeometry(100, 100, 800, 600)
        
        # 모델 파일 경로 저장 변수
        self.model_file_path = None
        
        # Whisper DLL 인스턴스 생성
        try:
            # 실행 파일이 있는 디렉토리 경로
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # DLL 파일 찾기
            possible_dll_names = ["whisper.dll", "whisper_cpp.dll", "whisper-cpp.dll", "whisperdll.dll"]
            dll_path = find_dll_file(current_dir, possible_dll_names)
            
            if not dll_path:
                raise Exception("Whisper DLL 파일을 찾을 수 없습니다.")
            
            # Vulkan 지원 여부 확인을 위한 체크박스 상태 초기화
            self.vulkan_enabled = True  # 기본값: Vulkan 사용
            self.whisper = WhisperDLL(dll_path=dll_path, vulkan_support=self.vulkan_enabled)
            self.model_loaded = False
        
        except Exception as e:
            QMessageBox.critical(self, "오류", f"Whisper DLL 초기화 실패: {str(e)}")
            self.whisper = None
        
        # 스레드 초기화
        self.recording_thread = None
        self.transcription_thread = None
        
        # 모델 디렉토리 생성
        self.models_dir = os.path.join(current_dir, "models")
        if not os.path.exists(self.models_dir):
            try:
                os.makedirs(self.models_dir)
            except Exception as e:
                print(f"모델 디렉토리 생성 실패: {str(e)}")
        
        # UI 초기화
        self.init_ui()
    
    def init_ui(self):
        """UI 구성 요소 초기화"""
        # 메인 위젯 및 레이아웃
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 폰트 설정
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        
        # 타이틀 라벨
        title_label = QLabel("Whisper 다국어 음성 인식 프로그램")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(title_label)
        
        # 모델 섹션
        model_layout = QHBoxLayout()
        model_label = QLabel("Whisper 모델:")
        self.model_path_label = QLabel("모델이 로드되지 않음")
        self.model_path_label.setStyleSheet("color: red;")
        
        load_model_btn = QPushButton("모델 로드")
        load_model_btn.clicked.connect(self.load_model)
        
        # 모델 다운로드 버튼 추가
        download_model_btn = QPushButton("모델 다운로드")
        download_model_btn.clicked.connect(self.show_model_downloader)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_path_label, 1)
        model_layout.addWidget(download_model_btn)
        model_layout.addWidget(load_model_btn)
        main_layout.addLayout(model_layout)
        
        # Vulkan 사용 여부 선택
        vulkan_layout = QHBoxLayout()
        vulkan_label = QLabel("가속 모드:")
        self.vulkan_checkbox = QComboBox()
        self.vulkan_checkbox.addItem("Vulkan GPU 가속", True)
        self.vulkan_checkbox.addItem("CPU 모드", False)
        self.vulkan_checkbox.currentIndexChanged.connect(self.toggle_vulkan)
        
        # ggml-vulkan.dll 파일이 존재하는지 확인
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ggml_vulkan_path = os.path.join(current_dir, "ggml-vulkan.dll")
        
        # 필요한 DLL 파일들의 존재 확인 및 상태 메시지 설정
        self.dll_status_label = QLabel()
        
        required_dlls = ["ggml.dll", "ggml-base.dll", "ggml-cpu.dll"]
        missing_dlls, all_found = check_required_dlls(current_dir, required_dlls)
        
        if missing_dlls:
            self.dll_status_label.setText(f"경고: 일부 DLL 파일이 없습니다: {', '.join(missing_dlls)}")
            self.dll_status_label.setStyleSheet("color: orange;")
        else:
            self.dll_status_label.setText("모든 필수 DLL 파일이 로드되었습니다.")
            self.dll_status_label.setStyleSheet("color: green;")
        
        # 파일이 없으면 Vulkan 옵션 비활성화
        if not os.path.exists(ggml_vulkan_path):
            self.vulkan_checkbox.setCurrentIndex(1)  # CPU 모드 선택
            self.vulkan_checkbox.setEnabled(False)  # 변경 불가능하게 설정
            self.vulkan_enabled = False
            
            # 툴팁 설정
            self.vulkan_checkbox.setToolTip("ggml-vulkan.dll 파일이 없어 GPU 가속을 사용할 수 없습니다.")
        
        vulkan_layout.addWidget(vulkan_label)
        vulkan_layout.addWidget(self.vulkan_checkbox)
        main_layout.addLayout(vulkan_layout)
        main_layout.addWidget(self.dll_status_label)
        
        # 언어 선택
        language_layout = QHBoxLayout()
        language_label = QLabel("인식 언어:")
        self.language_combo = QComboBox()
        
        # 주요 언어 추가
        languages = [
            ("자동 감지", "auto"),
            ("한국어", "ko"),
            ("영어", "en"),
            ("일본어", "ja"),
            ("중국어", "zh"),
            ("독일어", "de"),
            ("프랑스어", "fr"),
            ("스페인어", "es"),
            ("이탈리아어", "it"),
            ("러시아어", "ru")
        ]
        
        for name, code in languages:
            self.language_combo.addItem(name, code)
        
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        main_layout.addLayout(language_layout)
        
        # 녹음 버튼
        recording_layout = QHBoxLayout()
        self.record_btn = QPushButton("녹음 시작")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)  # 모델 로드 전에는 비활성화
        
        self.file_btn = QPushButton("파일 선택")
        self.file_btn.clicked.connect(self.select_audio_file)
        self.file_btn.setEnabled(False)  # 모델 로드 전에는 비활성화
        
        recording_layout.addWidget(self.record_btn)
        recording_layout.addWidget(self.file_btn)
        main_layout.addLayout(recording_layout)
        
        # 진행 상황 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")  # 퍼센트 표시 포맷 설정
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignRight)  # 텍스트 가운데 정렬
        main_layout.addWidget(self.progress_bar)
        
        # 결과 텍스트 영역
        result_label = QLabel("인식 결과:")
        main_layout.addWidget(result_label)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        main_layout.addWidget(self.result_text)
        
        # 버튼 섹션
        button_layout = QHBoxLayout()
        
        clear_btn = QPushButton("지우기")
        clear_btn.clicked.connect(self.clear_results)
        
        copy_btn = QPushButton("복사")
        copy_btn.clicked.connect(self.copy_results)
        
        save_btn = QPushButton("저장")
        save_btn.clicked.connect(self.save_results)
        
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(copy_btn)
        button_layout.addWidget(save_btn)
        main_layout.addLayout(button_layout)
        
    def show_model_downloader(self):
        """모델 다운로드 다이얼로그 표시"""
        downloader = ModelDownloader(self)
        downloader.exec()  # 모달 대화상자로 실행
    
    def on_model_downloaded(self, model_path):
        """모델이 다운로드된 후 호출되는 메서드"""
        if not model_path or not os.path.exists(model_path):
            return
        
        # 새로 다운로드된 모델을 현재 모델로 설정
        self.model_file_path = model_path
        self.model_path_label.setText(os.path.basename(model_path))
        self.model_path_label.setStyleSheet("color: green;")
        self.model_loaded = False  # 모델은 아직 메모리에 로드하지 않음
        
        # 버튼 활성화
        self.record_btn.setEnabled(True)
        self.file_btn.setEnabled(True)
        
        # 상태 메시지 표시
        QMessageBox.information(self, "모델 준비 완료", 
                                f"모델 파일이 다운로드되어 준비되었습니다.\n{os.path.basename(model_path)}")
        
    def toggle_vulkan(self, index):
        """가속 모드 토글"""
        self.vulkan_enabled = self.vulkan_checkbox.currentData()
        
        # 모델이 이미 로드된 경우, 다시 로드하도록 안내
        if self.model_loaded:
            QMessageBox.information(
                self, 
                "재로드 필요", 
                "가속 모드 설정을 변경했습니다. 설정을 적용하려면 모델을 다시 로드하세요."
            )
    
    def load_model(self):
        """Whisper 모델 파일 선택 및 유효성 확인"""
        if not self.whisper:
            QMessageBox.warning(self, "오류", "Whisper DLL이 초기화되지 않았습니다.")
            return
        
        file_dialog = QFileDialog()
        # 기본 시작 디렉토리를 모델 폴더로 설정
        start_dir = self.models_dir if os.path.exists(self.models_dir) else ""
        
        model_path, _ = file_dialog.getOpenFileName(
            self, "Whisper 모델 파일 선택", start_dir, 
            "모델 파일 (*.bin *.ggml *.en *.bin.* *.*);;모든 파일 (*.*)"
        )
        
        if model_path:
            try:
                self.progress_bar.setRange(0, 0)  # 진행 상황 표시
                self.progress_bar.setFormat("모델 확인 중... %p%")
                QApplication.processEvents()  # UI 업데이트
                
                # 현재 Vulkan 설정으로 새 WhisperDLL 인스턴스 생성
                current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                dll_path = None
                possible_dll_names = ["whisper.dll", "whisper_cpp.dll", "whisper-cpp.dll", "whisperdll.dll"]
                for name in possible_dll_names:
                    path = os.path.join(current_dir, name)
                    if os.path.exists(path):
                        dll_path = path
                        break
                
                if dll_path is None:
                    raise Exception("Whisper DLL 파일을 찾을 수 없습니다.")
                
                self.whisper = WhisperDLL(dll_path=dll_path, vulkan_support=self.vulkan_enabled)
                
                # 모델 파일 유효성만 확인 (실제 로드는 하지 않음)
                self.whisper.check_model_validity(model_path)
                
                # 모델 파일 경로 저장
                self.model_file_path = model_path
                
                # 진행 상황 표시 복원
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(100)
                self.progress_bar.setFormat("%p%")
                
                # 성공 시 UI 업데이트
                self.model_path_label.setText(os.path.basename(model_path))
                self.model_path_label.setStyleSheet("color: green;")
                self.model_loaded = False  # 모델은 아직 메모리에 로드하지 않음
                
                # 가속 모드 상태 표시
                acceleration_text = self.whisper.acceleration_mode if hasattr(self.whisper, 'acceleration_mode') else "CPU"
                self.model_path_label.setText(f"{os.path.basename(model_path)} ({acceleration_text} 모드) - 파일 확인 완료")
                
                # 버튼 활성화
                self.record_btn.setEnabled(True)
                self.file_btn.setEnabled(True)
                
                QMessageBox.information(self, "성공", "모델 파일이 유효합니다. 음성 인식 시 모델이 로드됩니다.")
            except Exception as e:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("%p%")
                
                error_msg = str(e)
                QMessageBox.critical(self, "오류", f"모델 파일 확인 실패: {error_msg}")
                
                # 로그 출력
                print(f"모델 파일 확인 실패: {error_msg}")
                import traceback
                traceback.print_exc()
    
    def toggle_recording(self):
        """녹음 시작/중지 전환"""
        if not self.model_file_path:
            QMessageBox.warning(self, "오류", "먼저 Whisper 모델 파일을 선택하세요.")
            return
        
        if self.recording_thread and self.recording_thread.is_recording:
            # 녹음 중지
            self.recording_thread.stop()
            self.record_btn.setText("녹음 시작")
            self.file_btn.setEnabled(True)
        else:
            # 녹음 시작
            self.record_btn.setText("녹음 중지")
            self.file_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.result_text.clear()
            
            # 녹음 스레드 생성 및 시작
            self.recording_thread = RecordingThread()
            self.recording_thread.update_progress.connect(self.update_progress)
            self.recording_thread.finished.connect(self.on_recording_finished)
            self.recording_thread.start()
    
    def update_progress(self, value):
        """진행 상황 업데이트"""
        self.progress_bar.setValue(value)
    
    def on_recording_finished(self, audio_file):
        """녹음 완료 후 처리"""
        self.record_btn.setText("녹음 시작")
        self.file_btn.setEnabled(True)
        
        # 녹음된 파일로 인식 시작
        self.transcribe_audio(audio_file)
    
    def select_audio_file(self):
        """오디오 파일 선택"""
        if not self.model_file_path:
            QMessageBox.warning(self, "오류", "먼저 Whisper 모델 파일을 선택하세요.")
            return
        
        file_dialog = QFileDialog()
        audio_file, _ = file_dialog.getOpenFileName(
            self, "오디오 파일 선택", "", "오디오 파일 (*.wav *.mp3 *.ogg);;모든 파일 (*.*)"
        )
        
        if audio_file:
            self.transcribe_audio(audio_file)
    
    def transcribe_audio(self, audio_file):
        """오디오 파일 인식"""
        if not self.model_file_path:
            QMessageBox.warning(self, "오류", "먼저 Whisper 모델 파일을 선택하세요.")
            return
        
        # 선택된 언어 가져오기
        language = self.language_combo.currentData()
        if language == "auto":
            language = None
        
        # UI 업데이트
        self.progress_bar.setValue(0)
        self.result_text.clear()
        self.result_text.setPlaceholderText("모델 로드 중...")
        
        # 이 시점에서 모델을 실제로 로드 (필요한 경우)
        if not self.model_loaded or not hasattr(self.whisper, 'ctx') or not self.whisper.ctx:
            try:
                # 기존 모델이 있으면 먼저 해제
                self.unload_model()
                
                self.progress_bar.setRange(0, 0)  # 불확정 진행 상황 모드
                QApplication.processEvents()  # UI 업데이트
                
                # 모델 로드
                self.whisper.load_model(self.model_file_path)
                self.model_loaded = True
                
                # 로드 성공 표시
                self.model_path_label.setText(f"{os.path.basename(self.model_file_path)} ({self.whisper.acceleration_mode} 모드) - 로드됨")
            except Exception as e:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                self.result_text.setPlaceholderText("")
                QMessageBox.critical(self, "모델 로드 오류", str(e))
                return
        
        # 이제 로드된 모델로 인식 시작
        self.result_text.setPlaceholderText("인식 중...")
        
        # 파일 인식 시작
        self.transcription_thread = TranscriptionThread(self.whisper, audio_file, language)
        self.transcription_thread.finished.connect(self.on_transcription_finished)
        self.transcription_thread.progress.connect(self.on_transcription_progress)  # 실시간 텍스트 업데이트 연결
        self.transcription_thread.progress_percent.connect(self.update_transcription_progress)  # 진행률 업데이트 연결
        self.transcription_thread.error.connect(self.on_transcription_error)
        self.transcription_thread.start()
        
        # 진행 상황 표시 - 0-100% 범위로 설정
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("인식 중... %p%")
    
    def on_transcription_progress(self, text):
        """인식 진행 상황 텍스트 업데이트 (실시간)"""
        self.result_text.setPlaceholderText("")
        self.result_text.setText(text)

    def update_transcription_progress(self, percent):
        """인식 진행률 업데이트 (실시간)"""
        self.progress_bar.setValue(percent)
        if percent < 100:
            self.progress_bar.setFormat(f"인식 중... %p%")
        else:
            self.progress_bar.setFormat("%p%")

    def on_transcription_finished(self, text):
        """인식 완료 후 처리"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("%p%")  # 완료시 퍼센트 표시로 복귀
        self.result_text.setPlaceholderText("")
        self.result_text.setText(text)
        
    def on_transcription_error(self, error_msg):
        """인식 오류 처리"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")  # 오류 발생 시 퍼센트 표시로 복귀
        self.result_text.setPlaceholderText("")
        QMessageBox.critical(self, "인식 오류", error_msg)
        
        self.unload_model()
        
    def unload_model(self):
        """모델을 메모리에서 해제"""
        try:
            if self.whisper and hasattr(self.whisper, 'ctx') and self.whisper.ctx:
                # 새 free_model 메서드 사용
                self.whisper.free_model()
                self.model_loaded = False
                
                # UI 업데이트
                if self.model_file_path:
                    self.model_path_label.setText(f"{os.path.basename(self.model_file_path)} - 준비됨 (메모리 해제됨)")
        except Exception:
            pass
        
    def clear_results(self):
        """결과 지우기"""
        self.result_text.clear()
        self.progress_bar.setValue(0)
    
    def copy_results(self):
        """결과 클립보드에 복사"""
        text = self.result_text.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            QMessageBox.information(self, "복사 완료", "텍스트가 클립보드에 복사되었습니다.")
    
    def save_results(self):
        """결과 파일로 저장"""
        text = self.result_text.toPlainText()
        if not text:
            QMessageBox.warning(self, "저장 실패", "저장할 텍스트가 없습니다.")
            return
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "결과 저장", "", "텍스트 파일 (*.txt);;모든 파일 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                QMessageBox.information(self, "저장 완료", f"결과가 {file_path}에 저장되었습니다.")
            except Exception as e:
                QMessageBox.critical(self, "저장 실패", f"파일 저장 중 오류 발생: {str(e)}")
    
    def closeEvent(self, event):
        """프로그램 종료 시 처리"""
        try:
            # 녹음 중이면 중지
            if self.recording_thread and self.recording_thread.is_recording:
                self.recording_thread.stop()
            
            # 임시 파일 삭제
            if self.recording_thread and self.recording_thread.temp_file:
                try:
                    if os.path.exists(self.recording_thread.temp_file):
                        os.remove(self.recording_thread.temp_file)
                except Exception as e:
                    print(f"임시 파일 삭제 실패: {str(e)}")
            
            # Whisper 리소스 해제
            self.unload_model()
            
            print("프로그램 종료")
        except Exception as e:
            print(f"프로그램 종료 중 오류 발생: {str(e)}")
        
        event.accept()
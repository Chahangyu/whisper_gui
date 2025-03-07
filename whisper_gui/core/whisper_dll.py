"""
WhisperDLL class - Interface for the Whisper speech recognition DLL
"""

import os
import ctypes
import numpy as np

class WhisperDLL:
    def __init__(self, dll_path=None, vulkan_support=True):
        """Whisper DLL을 로드하고 필요한 함수를 설정합니다."""
        try:
            # GGML 관련 DLL 파일들 목록
            self.vulkan_support = vulkan_support
            self.extra_dlls = []
            self.acceleration_mode = "CPU"  # 기본값
            self.ctx = None  # 모델 컨텍스트 초기화
            
            # 현재 디렉토리 경로
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up two levels to get to the app root directory
            current_dir = os.path.dirname(os.path.dirname(current_dir))
            
            # whisper.dll 경로 설정
            if dll_path is None:
                dll_path = os.path.join(current_dir, "whisper.dll")
                if not os.path.exists(dll_path):
                    # 현재 디렉토리에서 whisper.dll을 찾지 못한 경우, 다른 이름 시도
                    possible_names = ["whisper.dll", "whisper_cpp.dll", "whisper-cpp.dll", "whisperdll.dll"]
                    found = False
                    for name in possible_names:
                        test_path = os.path.join(current_dir, name)
                        if os.path.exists(test_path):
                            dll_path = test_path
                            found = True
                            break
                    
                    if not found:
                        raise Exception(f"whisper.dll 파일을 찾을 수 없습니다. '{current_dir}' 디렉토리에 파일이 있는지 확인하세요.")
            
            # 필요한 DLL 파일 목록
            ggml_dlls = [
                "ggml.dll",
                "ggml-base.dll",
                "ggml-cpu.dll",
            ]
            
            # Vulkan 지원이 요청된 경우 목록에 추가
            if vulkan_support:
                ggml_dlls.append("ggml-vulkan.dll")
            
            # 모든 필요한 DLL 파일을 로드
            for dll_name in ggml_dlls:
                dll_path_full = os.path.join(current_dir, dll_name)
                if os.path.exists(dll_path_full):
                    try:
                        ggml_dll = ctypes.CDLL(dll_path_full)
                        self.extra_dlls.append(ggml_dll)
                        print(f"{dll_name} 로드 성공")
                        
                        # 가속 모드 설정
                        if dll_name == "ggml-vulkan.dll" and vulkan_support:
                            self.acceleration_mode = "Vulkan GPU"
                    except Exception as e:
                        print(f"경고: {dll_name} 로드 실패: {str(e)}")
            
            # 메인 Whisper DLL 로드
            print(f"whisper dll 로드 경로: {dll_path}")
            self.dll = ctypes.CDLL(dll_path)
            
            # 함수 초기화
            if not self.initialize():
                raise Exception("DLL 함수 초기화 실패")
                
        except Exception as e:
            print(f"WhisperDLL 초기화 실패: {str(e)}")
            raise
    
    def load_model(self, model_path):
        """Whisper 모델을 로드합니다."""
        if self.ctx:
            self.dll.whisper_free(self.ctx)
            self.ctx = None
        
        try:
            # 파일이 존재하는지 확인
            if not os.path.exists(model_path):
                raise Exception(f"모델 파일이 존재하지 않습니다: {model_path}")
            
            print(f"모델 파일 로드 시도: {model_path}")
            
            # 파일 경로를 바이트로 인코딩
            model_path_bytes = model_path.encode('utf-8')
            
            # 모델 로드 시도 (whisper_init_from_file 사용)
            self.ctx = self.dll.whisper_init_from_file(model_path_bytes)
            
            # 로드 실패 시
            if not self.ctx or self.ctx == 0:
                raise Exception("모델 초기화 실패: whisper_init_from_file이 null을 반환했습니다")
            
            print("모델 로드 성공")
            return True
        except Exception as e:
            print(f"모델 로드 오류: {str(e)}")
            if self.ctx:
                try:
                    self.dll.whisper_free(self.ctx)
                except:
                    pass
                self.ctx = None
            raise Exception(f"모델 로드 실패: {str(e)}")
            
    def check_model_validity(self, model_path):
        """모델 파일의 유효성만 검사합니다. 실제 로드는 하지 않습니다."""
        try:
            # 파일이 존재하는지 확인
            if not os.path.exists(model_path):
                raise Exception(f"모델 파일이 존재하지 않습니다: {model_path}")
            
            # 파일 크기 확인 (최소 크기 체크)
            file_size = os.path.getsize(model_path)
            if file_size < 1024:  # 1KB 미만은 유효한 모델 파일이 아닐 가능성이 높음
                raise Exception(f"모델 파일이 너무 작습니다 ({file_size} bytes). 손상된 파일일 수 있습니다.")
            
            print(f"모델 파일 확인: {model_path} ({file_size/1024/1024:.2f} MB)")
            
            # 모델 파일이 유효하다고 판단
            return True
        except Exception as e:
            print(f"모델 검사 오류: {str(e)}")
            raise Exception(f"모델 파일 검사 실패: {str(e)}")
    
    def transcribe(self, audio_data, language=None):
        """오디오 데이터를 텍스트로 변환합니다."""
        if not self.ctx:
            raise Exception("모델이 로드되지 않았습니다")
        
        try:
            # 오디오 데이터를 float32 배열로 변환
            audio_float = audio_data.astype(np.float32)
            
            # 배열을 C 포인터로 변환
            audio_ptr = audio_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # 기본 파라미터 가져오기
            params = self.dll.whisper_full_default_params()
            if not params:
                raise Exception("whisper_full_default_params가 NULL을 반환했습니다")
            
            # 변환 실행
            print(f"오디오 변환 시작 (길이: {len(audio_float)} 샘플)")
            result = self.dll.whisper_full(self.ctx, params, audio_ptr, len(audio_float))
            
            if result != 0:
                raise Exception(f"오디오 변환 실패: 코드 {result}")
            
            # 결과 텍스트 가져오기 (세그먼트 조합)
            text = ""
            n_segments = self.dll.whisper_full_n_segments(self.ctx)
            print(f"인식된 세그먼트 수: {n_segments}")
            
            for i in range(n_segments):
                segment_text = self.dll.whisper_full_get_segment_text(self.ctx, i)
                if segment_text:
                    text += segment_text.decode('utf-8', errors='replace') + " "
            
            return text.strip()
        except Exception as e:
            print(f"변환 오류: {str(e)}")
            raise
    
    def __del__(self):
        """객체 소멸 시 리소스 해제"""
        try:
            if hasattr(self, 'ctx') and self.ctx:
                self.dll.whisper_free(self.ctx)
                self.ctx = None
            print("WhisperDLL 리소스가 해제되었습니다.")
        except Exception as e:
            print(f"WhisperDLL 리소스 해제 중 오류 발생: {str(e)}")
            
    def initialize(self):
        """DLL 함수 초기화 및 설정"""
        try:
            # Whisper DLL 함수 설정
            # 초기화 및 해제 함수
            self.dll.whisper_init_from_file.argtypes = [ctypes.c_char_p]
            self.dll.whisper_init_from_file.restype = ctypes.c_void_p
            
            self.dll.whisper_free.argtypes = [ctypes.c_void_p]
            self.dll.whisper_free.restype = None
            
            # 전체 처리 파라미터 및 실행 함수
            self.dll.whisper_full_default_params.argtypes = []
            self.dll.whisper_full_default_params.restype = ctypes.c_void_p
            
            self.dll.whisper_full.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            self.dll.whisper_full.restype = ctypes.c_int
            
            # 세그먼트 관련 함수
            self.dll.whisper_full_n_segments.argtypes = [ctypes.c_void_p]
            self.dll.whisper_full_n_segments.restype = ctypes.c_int
            
            self.dll.whisper_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.whisper_full_get_segment_text.restype = ctypes.c_char_p
            
            print("Whisper DLL 함수가 초기화되었습니다.")
            return True
        except Exception as e:
            print(f"Whisper DLL 함수 초기화 실패: {str(e)}")
            return False
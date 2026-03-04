# qec_sim/core/logger.py
import os
import csv

class QECLogger:
    """학습 과정의 콘솔 출력, 텍스트 로그, 메트릭(CSV) 저장을 전담하는 클래스"""
    
    def __init__(self, csv_path: str = None):
        self.csv_path = csv_path
        self.txt_path = None
        self.history = []
        
        if self.csv_path:
            base_name, _ = os.path.splitext(self.csv_path)
            self.txt_path = f"{base_name}.txt"
            
            # 텍스트 로거 초기화
            os.makedirs(os.path.dirname(self.txt_path) or '.', exist_ok=True)
            with open(self.txt_path, 'w', encoding='utf-8') as f:
                f.write("=== QEC Training Log ===\n")

    def info(self, message: str, print_console: bool = True):
        """메시지를 콘솔에 출력하고 txt 파일에 기록합니다."""
        if print_console:
            print(message)
            
        if self.txt_path:
            with open(self.txt_path, 'a', encoding='utf-8') as f:
                f.write(message + "\n")

    def log_metrics(self, metrics: dict):
        """에포크별 평가 지표를 내부 리스트에 저장하고 CSV 파일로 덮어씁니다."""
        self.history.append(metrics)
        
        if self.csv_path:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                if not self.history:
                    return
                # 딕셔너리의 키를 CSV 헤더로 자동 사용
                writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
                writer.writeheader()
                writer.writerows(self.history)
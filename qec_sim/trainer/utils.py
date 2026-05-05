import os


def timestamped_output_dir(output_dir: str, timestamp: str) -> str:
    """`<timestamp>_<basename>` 형태로 부모 디렉토리 아래 새 경로를 만든다.

    예: output_dir='./results/foo', timestamp='20260505_213733'
        → './results/20260505_213733_foo'
    `ls`가 시간순으로 정렬되도록 timestamp를 prefix로 둔다.
    """
    return os.path.join(os.path.dirname(output_dir),
                        f"{timestamp}_{os.path.basename(output_dir)}")

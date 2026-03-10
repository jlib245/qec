# qec_sim/core/registry.py
from typing import Dict, Generic, List, Type, TypeVar

T = TypeVar('T')


class Registry(Generic[T]):
    """등록(register) + 조회(get) 패턴을 제공하는 범용 레지스트리."""

    def __init__(self, name: str):
        self._name = name
        self._store: Dict[str, Type[T]] = {}

    def register(self, key: str):
        """클래스 데코레이터. @registry.register("이름") 형태로 사용."""
        def decorator(cls: Type[T]) -> Type[T]:
            self._store[key] = cls
            return cls
        return decorator

    def get(self, key: str) -> Type[T]:
        """이름으로 등록된 클래스를 반환합니다."""
        if key not in self._store:
            raise KeyError(
                f"'{key}'이(가) {self._name} 레지스트리에 없습니다. "
                f"등록된 항목: {self.keys()}"
            )
        return self._store[key]

    def keys(self) -> List[str]:
        return list(self._store.keys())

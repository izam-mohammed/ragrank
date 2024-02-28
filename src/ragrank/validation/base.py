from abc import ABC, abstractclassmethod


class RagrankBaseValidation(ABC):

    @abstractclassmethod
    def validate(self, object): ...

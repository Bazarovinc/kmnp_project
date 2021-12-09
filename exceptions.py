class WrongAnswer(Exception):
    message = 'Введен некорректный ответ! Повторите'

    def __str__(self):
        return self.message

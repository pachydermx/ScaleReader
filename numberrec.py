
class NumberRecognizer:
    def __init__(self, args):
        self.input_parts = args
        self.dic = {
            0:[True, True, True, False, True, True, True],
            1:[False, False, True, False, False, True, False],
            2:[True, False, True, True, True, False, True],
            3:[True, False, True, True, False, True, True],
            4:[False, True, True, True, False, True, False],
            5:[True, True, False, True, False, True, True],
            6:[True, True, False, True, True, True, True],
            7:[True, False, False, False, False, True, True],
            8:[True, True, True, True, True, True, True],
            9:[True, True, True, True, False, True, True]
        }

    def recoginze(self):
        result = -1
        for key, value in self.dic.iteritems():
            if value == self.input_parts:
                result = key
                break
        return result


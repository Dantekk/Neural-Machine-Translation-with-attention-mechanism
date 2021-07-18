class DataLoader(object):

    def __init__(self, ):
        pass
    
    # Load ita/eng row data from .txt file
    @staticmethod
    def load(path_to_data):
        translation_file = open(path_to_data, "r", encoding='utf-8')
        raw_data = translation_file.read()
        translation_file.close()
        return raw_data

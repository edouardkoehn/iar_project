class Piece(object):
    """Represents single jigsaw puzzle piece.

    Each piece has identifier so it can be
    tracked across different individuals

    :param image: ndarray representing piece's RGB values
    :param index: Unique id within piece's parent image

    """

    def __init__(self, image, index, rotation=0):
        self.image = image[:]
        self.id = index
        self.rotation = rotation

    def __getitem__(self, index):
        return self.image.__getitem__(index)

    def size(self):
        """Returns piece size"""
        if self.rotation in [90, 270]:
            return self.image.shape[1]
        else:
            return self.image.shape[0]

    def shape(self):
        """Returns shape of piece's image"""
        return self.image.shape

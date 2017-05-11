from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping
import cv2

class VisualizeResult(Callback):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.image = cv2.imread('sample_image.png')
        self.ground_truth = self.makeLabelPretty(cv2.imread('sample_label.png'))

    # Accepts and returns a numpy array.
    def makeLabelPretty(label):
        prettyLabel = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
        colors = [
        [255,102,102],  # light red
        [255,255,102],  # light yellow
        [102,255,102],  # light green
        [102,255,255],  # light blue
        [102,102,255],  # light indigo
        [255,102,255],  # light pink
        [255,178,102],  # light orange
        [153,51,255],   # violet
        [153,0,0],      # dark red
        [153,153,0],    # dark yellow
        [0,102,0],      # dark green
        [0,76,153],     # dark blue
        [102,0,51],     # dark pink
        [0,153,153],    # dark turquoise
        ]
        num_classes = np.argmax(label,2)
        assert num_classes <= len(colors)

        for i in range(num_classes):
            


    def on_batch_end(self):
        print 

def getCallbacks(model_name='test.h5', num_classes=12, patience=12):
    checkpoint = ModelCheckpoint(
        model_name,
        monitor='val_loss',
        verbose=0,
        save_best_only=True)

    tb = TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_images=True)

    early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

    vis = VisualizeResult(num_classes)

    return [checkpoint, early, vis]

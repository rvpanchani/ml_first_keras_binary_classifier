from model import mini_conv
from preprocessor import train_data_generator,val_data_generator
from plotloss import PlotLosses

def train():
    batch_size = 10
    train_generator = train_data_generator(batch_size)
    val_generator = val_data_generator(batch_size)

    model = mini_conv()

    model.summary()

    #model.load_weights('weights/weights.h5')

    model.fit_generator(
        train_generator,
        steps_per_epoch=150 // batch_size,
        epochs=25,
        validation_data=val_generator,
        validation_steps=150 // batch_size,
        callbacks = [PlotLosses()]
    )

    model.save_weights('weights/weights.h5')


if __name__ == '__main__':
    train()
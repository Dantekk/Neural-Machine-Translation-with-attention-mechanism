from MachineTranslation import MachineTranslation

def model_training(mt):
    mt.build(attetion_mode=False)
    model_name = "model.h5"
    metrics_train, metrics_valid = mt.train(model_name = model_name)
    print("Train Accuracy = %.4f - Train Loss = %.4f" % (metrics_train[1], metrics_train[0]))
    print("Validation Accuracy = %.4f - Validation Loss = %.4f" % (metrics_valid[1], metrics_valid[0]))

def main():
    mt = MachineTranslation()

    mt.dataset_load()
    mt.split_dataset()

    model_training(mt)  # Model training


if __name__ == "__main__":
    main()

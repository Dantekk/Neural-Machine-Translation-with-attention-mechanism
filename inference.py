from MachineTranslation import MachineTranslation

def main():
    mt = MachineTranslation()
    mt.dataset_load()
    mt.split_dataset()
    mt.model_load("model.h5")

    typeInference = 2

    if typeInference == 1:
        mt.predict_sentence_target(12)
    elif typeInference == 2:
        attetion_mode = False
        mt.build_inference(attention_mode=attetion_mode)

        while True:
            sentence = input("Enter sentence to translate : ")
            pred_sentence = mt.predict_new_sentence(sentence, attention_mode=attetion_mode)
            print(pred_sentence)

main()

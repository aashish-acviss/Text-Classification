# Import the necessary functions and variables from model.py
from model import train_model, predict_category_m1, predict_category_m

# Train the model
train_model()

# Use the prediction function with custom text
custom_text = "leaveers certain level disappoint say man civilian read aliens abduct couple probe category aliens headlines comments meet america repulsive man november brick rivers house bad editor note wwn publish story read meet america repulsive man category headlines leave comment page honey angella katherine november november wwn staff fiery force reckon weekly world news proud introduce late read page honey angella katherine categories headlines comments delaware missing november november bernardo time authority notice trail go cold federal bureau investigation read delaware missing category headlines comment flatulence saves man kidnappers november november brick rivers thank god burrito explain black van roar read flatulence save man kidnappers category headlines comment post navigation old post page page page click news search weekly world news stories weekly world news stories man sells soul devil santa covid dracula goes vegan mob war avoided spelling bee mule elect mayor horoscopes occult paranormal aliens mutants ed anger wwn weekly world newsletter license privacy policy term use weekly world news build generatepress"

# predicted_label = predict_category_m1(custom_text)
predicted_label = predict_category_m(custom_text)

print(f"The predicted class label for the custom text is: {predicted_label}")

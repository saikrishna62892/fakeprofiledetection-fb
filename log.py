from flask import Flask, request, render_template
import pickle
import sexmachine.detector as gender

random_forest = pickle.load(open('random_forest.pkl', 'rb'))
support_vector = pickle.load(open('support_vector_machine.pkl', 'rb'))
naive_bayes = pickle.load(open('naive_bayes.pkl', 'rb'))

#Random forest prediction
rfr_prediction = random_forest.predict([11,23,35,46,57,68,79,80])
print rfr_prediction[0]
#support vector machine prediction
svm_prediction = support_vector.predict([11,23,35,46,57,68,79,80])
print svm_prediction[0]
#naive_bayes prediction
nvb_prediction = naive_bayes.predict([11,23,35,46,57,68,79,80])
print nvb_prediction[0]
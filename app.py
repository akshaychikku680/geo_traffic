from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

import gzip
import shutil

input_file = 'finalized_model.pkl.gz'
output_file = 'finalized_model.pkl'

with gzip.open(input_file, 'rb') as f_in:
    with open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load the trained model and label encoders
with open('finalized_model.pkl', 'rb') as model_file:  
    model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as encoder_file:  
    label_encoders = pickle.load(encoder_file)

district_data = {
    'THIRUVANANTHAPURAM CITY': ['Vattiyoorkavu', 'Vanchiyoor', 'Thumba', 'Kazhakkuttom', 'Thiruvallam',
                                'Nemom', 'Medical College', 'Poonthura', 'Peroorkada', 'Sreekariyam',
                                'Fort', 'Valiyathura', 'Poojappura', 'Museum', 'Vizhinjam', 'Thampanoor',
                                'Cantonment', 'Pettah', 'Kovalam', 'Mannanthala', 'Karamana'],
    'THIRUVANANTHAPURAM RURAL': ['Kadinamkulam', 'Varkala', 'Balaramapuram', 'Aruvikkara', 'Chirayinkil',
                                 'Kattakkada', 'Attingal', 'Neyyattinkara', 'Palode', 'Vellarada',
                                 'Pothencode', 'Vattappara', 'Maranalloor', 'Pozhiyoor',
                                 'Kallambalam', 'Poovar', 'Kilimanoor', 'Malayinkil', 'Mangalapuram',
                                 'Pangode', 'Nedumangad', 'Parassala', 'Naruvamoodu', 'Kanjiramkulam',
                                 'Marayamuttom', 'Ariyancode', 'Valiyamala', 'Venjarammood',
                                 'Pallickal', 'Kadakkavoor', 'Aryanad', 'Neyyardam', 'Nagarur',
                                 'Vilappilsala', 'Vithura', 'Ponmudi'],
    'KOLLAM CITY': ['Karunagappally PS', 'Kottiyam PS', 'Eravipuram', 'Paravoor PS', 'Kilikollur PS',
                    'Pallithottam PS', 'Kannanelure PS', 'Chavara PS', 'Sakthikulangara PS', 'Parippally PS',
                    'Anchalumoodu PS', 'Ochira PS', 'Chavara Thekkumbhagam', 'Kollam East PS', 'Kollam West PS',
                    'Chathannoor PS'],
    'KOLLAM RURAL': ['Kundara PS', 'Eroor PS', 'Puthoor PS', 'Anchal PS', 'Chithara (Valavupacha) PS',
                     'Punalur PS', 'Sooranad PS', 'Kottarakkara PS', 'Thenmala PS', 'Pooyappally PS',
                     'Pathanapuram PS', 'Kunnikode PS', 'East Kallada PS', 'Sasthamcotta PS', 'Kadakkal PS',
                     'Ezhukone PS', 'Kulathupuzha PS', 'Chadayamangalam PS'],
    'PATHANAMTHITTA': ['Chittar ','Enathu (old Kakkad)', 'Ranny ', 'Adoor ', 'Aranmula ',
                       'Konni ', 'Pandalam ', 'Elavumthitta PS', 'Perunadu', 'Kodumon ', 'Koipuram ',
                       'Vechoochira ', 'Perumpetty ', 'Pathanamthitta ', 'Malayalapuzha', 'Pulikeezhu ',
                       'Keezhvaipur ', 'Vadasserikkara', 'Koodal ', 'Thannithodu ', 'Pampa '],
    'ALAPPUZHA': ['Cherthala', 'Mannachery', 'Kareelakulangara', 'Alappuzha North', 'Pattanakkadu',
                  'Ambalapuzha', 'Nedumudy', 'Mannar', 'Kayamkulam', 'Muhamma', 'Kuthiathode PS',
                  'Kurathikad', 'Poochakkal', 'Aroor', 'Mavelikara', 'Kanakakunnu', 'Nooranadu',
                  'Harippad', 'Edathua', 'Mararikkulam', 'Pulincunnu', 'Alappuzha South', 'Punnapra',
                  'Vallikunnam', 'Thrikkunnapuzha', 'Chengannur', 'Arthinkal', 'Ramankari',
                  'Venmony', 'Veeyapuram'],
    'KOTTAYAM': ['Vaikom', 'Kuravilangadu', 'Ramapuram', 'Thidanadu', 'Gandhinagar', 'Pala',
                 'Ettumanoor', 'Pallikkathodu', 'Velloor', 'Vakathanam', 'Kumarakom', 'Chingavanam',
                 'Thalayolaparambu', 'Kaduthuruthy', 'Changanachery', 'Thrikkodithanam',
                 'Ponkunnam', 'Kidangoor', 'Erumely', 'Karukachal', 'Pampady', 'Mundakayam',
                 'Manarcadu', 'Ayarkunnam', 'Kanjirappally', 'Erattupetta', 'Kottayam East',
                 'Kottayam West', 'Manimala', 'Marangattupally', 'Melukavu'],
    'IDUKKI': ['Peerumedu', 'Thodupuzha', 'Thankamoney', 'Vandiperiyar', 'Munnar', 'Kanjikuzhy',
               'Santhanpara', 'Devikulam', 'Karimannoor', 'Karimkunnam', 'Rajakkadu', 'Nedumkandam',
               'Kumali', 'Vellathooval', 'Kattappana', 'Adimali', 'Kaliyar', 'Kanjar', 'Muttom',
               'Peruvanthanam', 'Vandanmedu', 'Cumbummettu', 'Upputhara',
               'Idukki', 'Kulamavu', 'Marayoor', 'Karimanal'],
     'ERNAKULAM CITY': [
        'Elamakkara', 'Hill Palace (Thrippunnithura)', 'Ernakulam Central ', 'Infopark', 'Palluruthy Kasaba',
        'Kadavanthra ', 'Palarivattom ', 'Ernakulam Town South ', 'Udayamperoor', 'Cheranelloor ', 'Mattancherry ',
        'Ambalamedu', 'Panangad ', 'Fort Kochi  ', 'Thrikkakara', 'Mulavukadu ', 'Kannamali ', 'Maradu', 'Kalamassery',
        'Ernakulam Town North ', 'Harbour', 'Thoppumpady ', 'Eloor '
    ],
    'ERNAKULAM RURAL': [
        'Njarakkal', 'Aluva West', 'North Parur ', 'Puthencruz', 'Oonnukal', 'Nedumbassery', 'Kuruppumpady',
        'Vadakkekara', 'Aluva', 'Muvattupuzha', 'Chottanikkara', 'Puthenvelikkara', 'Binanipuram', 'Vazhakulam',
        'Kunnathunadu', 'Angamaly', 'Kothamangalam', 'Munambam', 'Kalady', 'Perumbavoor', 'Kalloorkadu',
        'Kuttampuzha', 'Edathala', 'Thadiyittaparambu', 'Mulamthuruthy', 'Varapuzha', 'Kodanadu', 'Ramamangalam',
        'Pothanikkadu', 'Piravom', 'Kottapady', 'Ayyampuzha', 'Koothattukulam', 'Chengamanadu'
    ],
    'THRISSUR CITY': [
        'Thrissur Town East ', 'Viyyur', 'Chelakkara', 'Chavakkad', 'Kunnamkulam', 'Medical College PS ',
        'Wadakkanchery', 'Thrissur Town West ', 'Nedupuzha', 'Ollur', 'Peramangalam', 'Vadakkekkad', 'Guruvayoor',
        'Peechi', 'Guruvayoor Temple PS', 'Cheruthuruthy', 'Pavaratty', 'Mannuthy', 'Erumapetty', 'Pazhayannoor'
    ],
    'THRISSUR RURAL': [
        'Chalakkudy', 'Anthikad', 'Mala', 'Cherpu', 'Vellikulangara', 'Koratty', 'Valappad', 'Kodungallur',
        'Aloor', 'Kaipamangalam', 'Vadanappally', 'Kattoor', 'Irinjalakkuda', 'Mathilakam', 'Varantharappally',
        'Pudukkad', 'Kodakara', 'Athirappally/ Vettilappara'
    ],
    'PALAKKAD': [
        'Kongad', 'Shornur', 'Nemmara', 'Sreekrishnapuram','Kollengode',
        'Palakkad Town South', 'Chittur', 'Alathur', 'Vadakkenchery', 'Mannarkkad','Palakkad Town North',
        'Pattambi', 'Mankara', 'Walayar', 'Kozhinjampara','Nattukal', 'Agali ',
        'Koppam'
    ],
    'MALAPPURAM': [
        'Edakkara', 'Tirur', 'Kondotty', 'Kottakkal', 'Pandikkad','Nilambur',
        'Tanur', 'Parappanangadi','Karipur Air port  PS', 'Manjeri','Valanchery',
        'Malappuram','Kalikavu', 'Ponnani','Kolathur', 'Kuzhalmannam'
    ],
    'KOZHIKODE CITY': [
        'Feroke', 'Kunnamangalam', 'Mukkom', 'Medical College PS ','Vellayil', 'Kakkur ',
        'Elathur','Balussery ','Koduvally ',
        'Thamarassery'
    ], 
    
    'KOZHIKODE RURAL': ['Koyilandy ', 'Thamarassery', 'Koduvally ', 'Kuttiady ', 'Kakkur ', 'Atholi', 'Thiruvambady', 
                        'Meppayur ', 'Perambra ', 'Vatakara ', 'Mukkom', 'Payyoli', 'Thotilpalam', 'Edachery', 
                        'Valayam', 'Balussery', 'Chompala', 'Peruvannamoozhi', 'Kodenchery', 'Nadapuram ', 
                        'Koorachundu'],
    'WAYANAD': ['Ambalavayal', 'Noolpuzha', 'Kambalakkad', 'Mananthavadi', 'Sulthan Batheri', 'Meenangadi', 
                'Vythiri', 'Kalpetta', 'Meppadi', 'Padinjarethara', 'Vellamunda', 'Thondarnadu', 'Thirunelli', 
                'Panamaram', 'Pulpally', 'Thalappuzha', 'Kenichira'],
    'KANNUR CITY': ['Panoor', 'Edakkad', 'Chockly', 'New Mahe', 'Chakkarakkal', 'Kathirur', 'Valapattanam', 
                    'Mattannur', 'Dharmadam', 'Mayyil', 'Kuthuparamba', 'Kolavallur', 'Kannavam', 'Thalassery', 
                    'Pinarayi', 'Kannur Town ', 'Kannapuram'],
    'KASARAGOD': ['Bedakam', 'Hosdurg', 'Manjeshwar', 'Badiadka', 'Vellarikundu', 'Kasaragod', 'Adhur', 
                  'Cheemeni', 'Chittarikkal', 'Kumbla', 'Chandera', 'Nileshwar', 'Bekal', 'Vidyanagar', 
                  'Amabalathara', 'Melparamba PS', 'Rajapuram'],
    'KANNUR RURAL': ['Pariyaram MC PS', 'Payyavoor', 'Kelakam', 'Payyannur', 'Kudiyanmala', 'Alakode', 
                     'Taliparamba', 'Payangadi', 'Cherupuzha', 'Iritty', 'Peringome', 'Peravoor', 'Sreekandapuram', 
                     'Irikkur', 'Muzhakkunnu', 'Karikottakari', 'Ulikkal', 'Aralam', 'Maloor']    
}

@app.route('/')
def index():
    return render_template('index.html',district_data=district_data)

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/da')
def dat():
    data = pd.read_csv("combined_data.csv")
    return render_template("data.html",tables=[data.to_html()],titles=[''])

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    manual_input_data = {
        'Type Road': request.form['Type Road'],
        'Road Features': request.form['Road Features'],
        'Traffic Control': request.form['Traffic Control'],
        'Type Area':request.form['Type Area'],
        'Weather':request.form['Weather'],
        'Time Accident':request.form['Time Accident'],
        'Collision':request.form['Collision'],
        'District':request.form['District'],
        'PS Name':request.form['PS Name']
    }
    
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([manual_input_data])

    # Apply label encoding to categorical features
    for column in ['Type Road', 'Road Features', 'Traffic Control','Type Area','Weather','Time Accident','Collision','District','PS Name']:
        input_df[column] = label_encoders[column].transform(input_df[column])

    # Concatenate encoded categorical data and numerical data
    # new_data = input_df[['Type Road', 'Road Features', 'Traffic Control', 'Type Area','Weather','Time Accident','Collision','District','PS Name']]

    # Make prediction
    prediction = model.predict(input_df)

    # Interpret the prediction
    if prediction[0] == 3:
        result = "Fatal"
        return render_template('result.html', prediction=result)
    elif prediction[0] == 2:
        result = "Grevious Injury"
        return render_template('result2.html', prediction=result) 
    elif prediction[0] == 1:
        result = "Minor Injury"
        return render_template('result3.html', prediction=result)

    # Return the result
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)


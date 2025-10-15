from flask import Flask, render_template, request
from helper import create_vector_db, get_qa_chain

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        question = request.form['question']
        # print(question)
        
        chain = get_qa_chain()
        response = chain.invoke(question)['result']
        print(response)
        
        return render_template("index.html", question=question, response = response)

    else:
        return render_template("index.html")





if __name__ == "__main__":
    app.run(debug=True)
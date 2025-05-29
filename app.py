from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline
import torch
import re
from cleantext import clean
from better_profanity import profanity

app = Flask(__name__)

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device==0 else 'cpu'}\n")
summarizer = pipeline(
    'summarization',
    model='facebook/bart-large-cnn',
    tokenizer='facebook/bart-large-cnn',
    device=device
)

def pre_clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"Advertisement:?|Loading\.\.\.", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+", " ", text)
    text = clean(text,
                 fix_unicode=True,
                 no_urls=True,
                 no_emails=True,
                 no_phone_numbers=True,
                 lower=False)
    return text.strip()

def post_clean(summary: str) -> str:
    summary = re.sub(r"\s*\[\d+\]", "", summary)
    summary = re.sub(r"Advertisement:?|Loading\.\.\.", "", summary, flags=re.IGNORECASE)
    summary = re.sub(r"http\S+", "", summary)
    summary = profanity.censor(summary)
    summary = re.sub(r"\s{2,}", " ", summary).strip()
    if not summary.endswith(('.', '!', '?')):
        summary += '.'
    return summary

@app.route("/", methods=["GET"])
def index():
    return render_template_string("""
    <!doctype html>
    <html>
    <head><title>Document Summarization</title></head>
    <body>
      <h1>Document Summarization</h1>
      <form id="summarize-form">
        <textarea id="text-input" rows="10" placeholder="Enter text to summarize..."></textarea><br>
        <button type="submit">Summarize</button>
      </form>
      <div id="summary-output"></div>
      <script>
        document.getElementById('summarize-form').addEventListener('submit', async function(e) {
          e.preventDefault();
          const text = document.getElementById('text-input').value;
          const resp = await fetch('/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
          });
          const data = await resp.json();
          document.getElementById('summary-output').textContent = data.summary || ('Error: ' + data.error);
        });
      </script>
    </body>
    </html>
    """)

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        raw_text = request.get_json().get("text", "")
        cleaned = pre_clean(raw_text)

        result = summarizer(
            cleaned,
            max_length=200,
            min_length=80,
            do_sample=False,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True
        )
        summary = result[0]['summary_text']
        cleaned_summary = post_clean(summary)
        return jsonify({"summary": cleaned_summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

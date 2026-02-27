# 📚 Brightspace Course Assistant

A RAG chatbot embedded in Brightspace as a Content topic iframe. Each course has its own folder of uploaded content files — no authentication required, works for all students.

---

## How It Works

```
content/
├── course_names.txt          ← maps course IDs to display names
├── 297671/                   ← one folder per course (named by org unit ID)
│   ├── syllabus.pdf
│   ├── week1_notes.docx
│   └── policy.txt
└── 123456/
    └── ...
```

When a student opens the chatbot iframe, the app:
1. Reads the `?course_id=` URL parameter
2. Loads all files from `content/<course_id>/`
3. Chunks, embeds, and indexes them into ChromaDB
4. Answers questions grounded in that content only

---

## Setup

### 1. Create a GitHub repo and push all files

```
brightspace-chatbot/
├── app.py
├── requirements.txt
├── .gitignore
├── content/
│   ├── course_names.txt
│   └── 297671/
│       └── (your course files go here)
└── .streamlit/
    ├── config.toml
    └── secrets.toml.template
```

### 2. Add course content files

For each course:
1. Export content from Brightspace (PDFs, Word docs, HTML pages, text files)
2. Create a folder `content/<org_unit_id>/`
3. Upload your files into that folder
4. Add a line to `content/course_names.txt`:
   ```
   297671 = Data Management and Analytics — ITEC 3310
   ```

### 3. Deploy to Streamlit Community Cloud

1. Go to share.streamlit.io → **New app**
2. Point to your repo, branch `main`, file `app.py`
3. Click **Advanced settings → Secrets** and paste:
   ```toml
   GOOGLE_API_KEY = "AIza-your-google-key-here"
   ```
4. Deploy — note your app URL

### 4. Embed in Brightspace

For each course, create a Content topic with this HTML:

```html
<iframe
  src="https://YOUR-APP-URL.streamlit.app/?course_id=297671&embed=true"
  width="100%"
  height="700"
  frameborder="0"
  style="border-radius: 12px;">
</iframe>
```

Change `course_id=` to match the org unit ID of each course.

---

## Adding a New Course

1. Create folder `content/<new_course_id>/`
2. Upload content files into it
3. Add a line to `content/course_names.txt`
4. Commit and push to GitHub — Streamlit redeploys automatically
5. Create a new Content topic in the new Brightspace course with the correct `course_id` in the iframe URL

---

## Supported File Types

| Format | Supported |
|---|---|
| `.pdf` | ✅ |
| `.docx` | ✅ |
| `.txt` | ✅ |
| `.html` / `.htm` | ✅ |
| Other | Ignored |

---

## Updating Course Content

When course materials change:
1. Replace or add files in `content/<course_id>/`
2. Commit and push
3. Streamlit redeploys — the pipeline rebuilds automatically on next load

---

*Built for NBCC · 2026*

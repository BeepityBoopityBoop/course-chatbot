# 📚 Brightspace Course Assistant — RAG Chatbot

A RAG chatbot that connects live to Brightspace via the LE API, indexes course content files automatically, and answers student questions grounded in that content. Embedded as an iframe inside a Brightspace Content topic.

---

## Architecture

```
Brightspace Course (iframe in Content topic)
        │
        ▼
Streamlit App  ──► Brightspace LE API  ──► fetch content files
        │                                        │
        │          chunk → embed (MiniLM) → ChromaDB
        │
        └──► Gemini 2.5 Flash  ──► grounded answer
```

- **One app, many courses** — the course is identified by `?course_id=` in the URL
- **Auto-refresh** — content is re-indexed when the course ID changes
- **No assessment access** — quizzes, dropbox, surveys are explicitly excluded

---

## Setup

### 1. Create a new GitHub repo and push these files

```
brightspace-chatbot/
├── app.py
├── requirements.txt
├── .gitignore
└── .streamlit/
    ├── config.toml
    └── secrets.toml.template    ← safe to commit; actual secrets.toml is gitignored
```

### 2. Deploy to Streamlit Community Cloud

1. Go to share.streamlit.io → **New app**
2. Point to your repo, branch `main`, file `app.py`
3. Click **Advanced settings → Secrets** and paste:

```toml
GOOGLE_API_KEY   = "AIza-your-google-key-here"
BS_CLIENT_SECRET = "your-brightspace-client-secret-here"
```

4. Click **Deploy**. Note your app URL — e.g.:
   `https://YOUR_USERNAME-brightspace-chatbot-app-XXXX.streamlit.app`

---

## Embedding in Brightspace

Each course gets its own embed URL — just change the `course_id` parameter.

### Step 1 — Find the course org unit ID
Open the course in Brightspace. The URL will contain something like `/d2l/home/297671` — `297671` is the org unit ID.

### Step 2 — Build the embed URL
```
https://YOUR-APP-URL.streamlit.app/?course_id=297671
```

### Step 3 — Add to Brightspace as a Content topic

1. Open the course → **Content**
2. Navigate to the module where you want the chatbot
3. Click **New** → **Create a File** (or **Upload / Create → Create a File**)
4. Title it: `Course Assistant`
5. Switch the editor to **HTML source** (the `<>` button)
6. Paste this iframe code — replace the URL with your actual app URL:

```html
<iframe
  src="https://YOUR-APP-URL.streamlit.app/?course_id=297671"
  width="100%"
  height="700"
  frameborder="0"
  allow="clipboard-write"
  style="border-radius: 12px; border: 1px solid #2a3a60;">
</iframe>
```

7. Save and publish the topic

### Step 4 — Repeat for each course
For each new course, create a new Content topic using the same iframe HTML but with the correct `course_id`:
```
?course_id=123456   ← change this for each course
```

---

## Brightspace OAuth App Settings

| Setting | Value |
|---|---|
| Client ID | `2b9cbd14-1e83-4c45-beee-ac2d7f71ef84` |
| Instance | `https://nbcctest.brightspace.com` |
| Scopes | `content:file:read content:modules:read content:topics:read` |
| Grant type | Client Credentials |

---

## Supported File Types

| Format | Supported |
|---|---|
| `.txt` | ✅ |
| `.html` / `.htm` | ✅ |
| `.pdf` | ✅ |
| `.docx` | ✅ |
| Other | ⚠️ Attempted as plain text |
| Quizzes / Dropbox / Surveys | ❌ Excluded by design |

---

## Secrets Reference

| Secret key | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | aistudio.google.com → Get API key |
| `BS_CLIENT_SECRET` | Brightspace Admin → Manage Extensibility → OAuth 2.0 → your app |

---

*Built for NBCC — Brightspace RAG Integration · 2026*

# SHL AI Assessment Recommender

An AI-powered conversational recommendation system that helps users discover the most relevant SHL assessments based on job roles, skills, hiring requirements, and natural language queries.

Built with FastAPI, semantic search, and a lightweight frontend in a single deployable application.

---

## Features

- Conversational AI-based assessment recommendations
- Natural language query support
- Semantic similarity search
- SHL product catalog scraping and indexing
- FastAPI backend with REST APIs
- Integrated frontend served from the same deployment
- One-click deployment using Render
- Dockerized for easy hosting

---

## Project Structure

```bash
.
├── backend/
│   ├── app/
│   ├── data/
│   ├── scripts/
│   ├── requirements.txt
│   └── README.md
├── render.yaml
├── Dockerfile
└── README.md
```

---

## Tech Stack

- Python
- FastAPI
- BeautifulSoup
- Requests
- Sentence Transformers
- FAISS / Semantic Search
- HTML/CSS/JavaScript Frontend
- Docker
- Render

---

## How It Works

1. SHL catalog data is scraped and cleaned
2. Assessment metadata is embedded using semantic embeddings
3. User queries are converted into embeddings
4. Similarity search retrieves the most relevant assessments
5. Recommendations are returned conversationally

---

## Local Development

### Clone Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### Run Application

```bash
uvicorn backend.app.main:app --reload
```

Application runs at:

```bash
http://localhost:8000
```

---

## API Endpoints

### Health Check

```http
GET /health
```

### Recommend Assessments

```http
POST /recommend
```

Example Request:

```json
{
  "query": "Recommend assessments for backend developer hiring"
}
```

---

## SHL Catalog Scraper

The scraper collects assessment metadata directly from the SHL catalog.

Run scraper:

```bash
python backend/scripts/scrape_catalog.py
```

Output:

```bash
backend/data/catalog.json
```

---

## Deployment

### Deploy on Render

This project includes:

- `render.yaml`
- Docker configuration
- Production-ready FastAPI setup

Steps:

1. Push repository to GitHub
2. Create new Blueprint in Render
3. Connect repository
4. Deploy automatically

---

## Environment Variables

Example:

```env
PYTHON_VERSION=3.11
PORT=8000
```

---

## Future Improvements

- LLM-powered conversational memory
- Advanced reranking
- User authentication
- Analytics dashboard
- Multi-language support
- Resume-based recommendations

---

## License

MIT License

---

## Author

Built for the SHL AI Internship Assignment.
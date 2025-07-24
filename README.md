# AI Data Query Agent

An intelligent data analytics platform that converts natural language questions into SQL queries and generates automatic visualizations. Ask questions about your data in plain English and get instant insights with charts and tables.

## Features

- **Natural Language to SQL**: Convert plain English questions to SQL queries using AI
- **Dual LLM Support**: Choose between Google Gemini and Ollama for query generation
- **Automatic Visualizations**: Smart chart generation based on your data and questions
- **Multiple Interfaces**: Web UI and command-line interface
- **Real-time Results**: Instant query execution with formatted results
- **CSV Data Import**: Automatically convert CSV files to queryable database

## Demo

![AI Data Query Agent Interface](https://via.placeholder.com/800x400/4f46e5/ffffff?text=AI+Data+Query+Agent)

## Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key (optional: Ollama for local LLM)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ai-data-query-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
   ```

4. **Load your CSV data**
   ```bash
   # Place CSV files in the data/ directory
   python csv_to_sql.py
   ```

5. **Start the services**
   ```bash
   # Terminal 1: Start FastAPI backend
   uvicorn fastapi_app:app --reload --port 8000
   
   # Terminal 2: Start Flask frontend
   python flask_app.py
   ```

6. **Open your browser**
   ```
   http://localhost:5000
   ```

## Usage

### Web Interface

1. Navigate to `http://localhost:5000`
2. Enter your question in natural language
3. Select your preferred AI provider (Gemini/Ollama)
4. Toggle visualization if you want charts
5. Click Submit to get results

**Example Questions:**
- "Show me total sales by product"
- "What are the top 5 products by ad spend?"
- "Compare ad sales vs total sales for each item"
- "Which products have the highest click-through rate?"

### Command Line Interface

```bash
# Basic query
python ai_agent.py "Show me all products with sales over 1000"

# With visualization
python ai_agent.py "Top 10 products by revenue" --visualize
```

### API Usage

```python
import requests

# Query the API directly
response = requests.post("http://localhost:8000/query", json={
    "question": "Show me total sales by product",
    "visualize": True,
    "provider": "gemini"
})

result = response.json()
print(result["sql"])  # Generated SQL
print(result["table"])  # Query results
```

## Data Structure

The system expects CSV files in the `data/` directory. Current schema includes:

### Ad Sales & Metrics
- `date`, `item_id`, `ad_sales`, `impressions`, `ad_spend`, `clicks`, `units_sold`

### Product Eligibility
- `eligibility_datetime_utc`, `item_id`, `eligibility`, `message`

### Total Sales & Metrics
- `date`, `item_id`, `total_sales`, `total_units_ordered`

## API Endpoints

### FastAPI Backend (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check and database status |
| `/test-sql` | GET | Test SQL generation functionality |
| `/query` | POST | Main query endpoint for natural language questions |

### Request Format

```json
{
  "question": "Your natural language question",
  "visualize": true,
  "provider": "gemini"
}
```

### Response Format

```json
{
  "sql": "SELECT * FROM products...",
  "table": [{"column1": "value1", "column2": "value2"}],
  "chart_base64": "base64_encoded_chart_image"
}
```

## Configuration

### Environment Variables

Create a `.env` file with:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### LLM Providers

**Google Gemini (Recommended)**
- Requires API key from Google AI Studio
- Model: `gemini-2.5-flash`
- Best performance for SQL generation

**Ollama (Local)**
- Requires Ollama installation
- Model: `qwen2.5:7b`
- Runs locally, no API key needed
- Install: [Ollama Installation Guide](https://ollama.ai)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Flask Web UI  │───▶│  FastAPI Backend │───▶│   SQLite DB     │
│   (Port 5000)   │    │   (Port 8000)    │    │   (data.db)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  LangChain +     │
                       │  LLM (Gemini/    │
                       │  Ollama)         │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Plotly Charts   │
                       │  (Visualization) │
                       └──────────────────┘
```

## Development

### Project Structure

```
ai-data-query-agent/
├── data/                          # CSV data files
├── templates/                     # HTML templates
│   └── index.html
├── fastapi_app.py                # Main API server
├── flask_app.py                  # Web frontend
├── ai_agent.py                   # CLI interface
├── csv_to_sql.py                 # Data loader
├── requirements.txt              # Dependencies
├── .env                          # Environment variables
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

### Adding New Data

1. Place CSV files in the `data/` directory
2. Run `python csv_to_sql.py` to update the database
3. Restart the FastAPI server

### Customizing Charts

The system automatically selects chart types based on:
- Data types (numeric vs categorical)
- Number of columns
- Question context

Chart types supported:
- Bar charts for categorical data
- Line charts for time series
- Scatter plots for correlations
- Pie charts for proportions

## Troubleshooting

### Common Issues

**"No module named 'langchain'"**
```bash
pip install -r requirements.txt
```

**"API key not found"**
- Check your `.env` file exists
- Verify `GOOGLE_API_KEY` is set correctly
- Restart the FastAPI server

**"Database not found"**
```bash
python csv_to_sql.py
```

**Charts not generating**
- Install kaleido: `pip install kaleido`
- Check if data has numeric columns for visualization

### Health Check

Visit `http://localhost:8000/health` to verify:
- Database connection
- Available tables
- API key status
- Sample data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://langchain.com/) for LLM integration
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Plotly](https://plotly.com/) for visualization
- [Google Gemini](https://ai.google.dev/) for AI capabilities

## Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Visit the health check endpoint: `http://localhost:8000/health`
3. Open an issue on GitHub
4. Review the logs in your terminal

---

**Made with ❤️ for data enthusiasts**
# SupportSense

SupportSense is a Python project for exploring and analyzing customer-support
tickets. The repository currently provides the project foundation, a sample
dataset, and a smoke test that can be extended as the analysis and application
features are developed.

## Dataset

The included [`data/support_tickets.csv`](data/support_tickets.csv) file contains
300 synthetic support tickets. Each record has the following fields:

| Field | Description |
| --- | --- |
| `ticket_id` | Unique ticket identifier |
| `created_at` | Date and time the ticket was created |
| `channel` | Origin of the request, such as email, chat, mobile app, or web form |
| `region` | Customer region represented by a country code |
| `text` | Customer's support request |
| `category` | Support topic assigned to the ticket |
| `priority` | Ticket urgency (`low`, `medium`, or `high`) |
| `sentiment` | Sentiment label (`negative`, `neutral`, or `positive`) |

> **Note:** The bundled data is intended for development and demonstration. Do
> not treat it as production customer data or use it to draw real-world business
> conclusions.

## Repository structure

```text
SupportSense/
├── data/
│   └── support_tickets.csv   # Sample support-ticket data
├── src/
│   └── support_sense/        # Python package
├── tests/
│   └── test_smoke.py         # Initial test suite
├── .env.example              # Template for future environment variables
├── pyproject.toml            # Project and tool configuration
└── requirements.txt          # Pinned Python dependencies
```

## Requirements

- Python 3.10 or newer
- `pip`

## Getting started

1. Clone the repository and enter the project directory:

   ```bash
   git clone <repository-url>
   cd SupportSense
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, activate it with:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the pinned dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Development

Run the test suite from the repository root:

```bash
python -m pytest
```

Lint the code with Ruff:

```bash
python -m ruff check .
```

Check formatting without modifying files:

```bash
python -m ruff format --check .
```

Apply Ruff formatting when needed:

```bash
python -m ruff format .
```

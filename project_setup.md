# Project Setup Guide

This guide will walk you through setting up the project, including configuring the PostgreSQL database, Redis server, and running the frontend and backend servers.

## Prerequisites

- **Python 3.11**
- **PostgreSQL**
- **Redis**
- **Node.js and pnpm (for frontend)**
- **Make** (optional, for running migrations)
- **Conda** (for Python environment management)

---

## PostgreSQL Database Setup

1. **Install PostgreSQL** if you haven't already.

2. **Create a new user and database**:

   ```bash
   psql -d postgres -U postgres
   ```

   In the PostgreSQL shell, run:

   ```sql
   CREATE USER myuser WITH PASSWORD 'mypassword';
   CREATE DATABASE mydb;
   GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;
   ```

3. **Install `pgvector` extension**:

   ```bash
   cd /tmp
   git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
   cd pgvector/
   make
   sudo make install
   ```

   *Note: Ensure that `pg_config` is in your system's PATH. You can check its location with:*

   ```bash
   which pg_config
   ```

4. **Create the `vector` extension** in your database:

   ```sql
   CREATE EXTENSION vector;
   ```

5. **Add the `array_to_vector` function** to your PostgreSQL database:

   ```sql
   CREATE OR REPLACE FUNCTION array_to_vector(arr float[]) RETURNS vector AS $$
   BEGIN
       RETURN arr::vector;
   END;
   $$ LANGUAGE plpgsql IMMUTABLE;
   ```

6. **Verify the `array_to_vector` function exists**:

   ```sql
   \df+ array_to_vector
   ```

---

## Redis Setup

1. **Start the Redis server**:

   ```bash
   redis-server
   ```

---

## Development Setup

### 1. Install Dependencies

- **Create a Conda Environment**:

  Install Conda if you haven't already. You can use [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

  ```bash
  conda create -n myenv python=3.11
  ```

  Replace `myenv` with your preferred environment name.

- **Activate the Conda Environment**:

  ```bash
  conda activate myenv
  ```

- **Navigate to the project directory** (if not already there):

  ```bash
  cd backend
  ```

- **Install Project Dependencies**:

  ```bash
  pip install -r requirements.lock
  ```

### 2. Prepare Environment

Ensure your Conda environment is activated before proceeding.

- **Copy the example environment variables file**:

  ```bash
  cp .env.example .env
  ```

- **Edit `.env`** to set environment variables:

  ```env
  ENVIRONMENT=local

  # Database Configuration
  TIDB_HOST=localhost
  TIDB_PORT=5432
  TIDB_USER=myuser
  TIDB_PASSWORD=mypassword
  TIDB_DATABASE=mydb
  TIDB_SSL=False

  # Secret Key (generate a new one)
  SECRET_KEY=your_generated_secret_key_here
  ```

  *Note: You can generate a new secret key by running:*

  ```bash
  python3 -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

  *Ensure that the `SECRET_KEY` is at least 32 characters long and keep it secure.*

### 3. Run Migrations

Ensure your Conda environment is activated before running migrations.

- **Navigate to the backend directory** (if not already there):

  ```bash
  cd backend
  ```

- **Run database migrations**:

  ```bash
  make migrate
  ```

### 4. Run Development Server

Ensure your Conda environment is activated before starting the backend server.

- **Activate the Conda environment** (if not already activated):

  ```bash
  conda activate myenv
  ```

- **Start the backend server**:

  ```bash
  rye run python main.py runserver
  ```

### 5. Create Admin User

Ensure your Conda environment is activated before running the bootstrap script.

- **Run the bootstrap script to create credentials for the admin user**:

  ```bash
  python bootstrap.py
  ```

---

## Running the Project

### Frontend

- **Navigate to the frontend directory**:

  ```bash
  cd frontend
  ```

- **Install frontend dependencies**:

  ```bash
  pnpm install
  ```

- **Run the frontend development server**:

  ```bash
  pnpm dev
  ```

### Backend

Ensure your Conda environment is activated before starting the backend server.

- **Navigate to the backend directory**:

  ```bash
  cd backend
  ```

- **Activate the Conda environment** (if not already activated):

  ```bash
  conda activate myenv
  ```

- **Start the backend server**:

  ```bash
  rye run python main.py runserver
  ```

### Celery Worker

Ensure your Conda environment is activated before starting the Celery worker.

- **Start the Celery worker**:

  ```bash
  celery -A app.celery worker -l DEBUG
  ```

### Celery Flower Monitoring

Ensure your Conda environment is activated before starting Celery Flower.

- **Start Celery Flower to monitor tasks**:

  ```bash
  celery -A app.celery flower --port=5555
  ```

---

## Additional Notes

- **Environment Variables**: Ensure all required environment variables are correctly set in your `.env` file.
- **Database Functions**: The `array_to_vector` function is crucial for vector operations in the database.
- **Redis Server**: Make sure the Redis server is running before starting the backend to handle background tasks.
- **Conda Environment**: Always ensure that your Conda environment is activated when working on the backend to use the correct Python packages.

---

## Quick Commands Summary

- **Start Redis Server**:

  ```bash
  redis-server
  ```

- **Create and Activate Conda Environment**:

  ```bash
  conda create -n myenv python=3.11
  conda activate myenv
  ```

- **Install Dependencies**:

  ```bash
  pip install -r requirements.lock
  ```

- **Run Migrations**:

  ```bash
  make migrate
  ```

- **Start Backend Server**:

  ```bash
  rye run python main.py runserver
  ```

- **Start Frontend Server**:

  ```bash
  pnpm dev
  ```

- **Start Celery Worker**:

  ```bash
  celery -A app.celery worker -l DEBUG
  ```

- **Start Celery Flower**:

  ```bash
  celery -A app.celery flower --port=5555
  ```

---

## Generating Secret Key

Generate a secure secret key for your application:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Contact and Support

If you encounter any issues during setup, please consult the project's documentation or reach out to the development team for assistance.

---

**Enjoy working with the project!**

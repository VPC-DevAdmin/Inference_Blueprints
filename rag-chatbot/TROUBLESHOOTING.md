# Troubleshooting Guide

This document contains all common issues encountered during development and their solutions.

## Table of Contents

- [API Common Issues](#api-common-issues)
- [UI Common Issues](#ui-common-issues)

### API Common Issues

#### "Gateway authentication not configured"

**Solution**:

1. Create a `.env` file in the `api` directory.
2. Configure either Keycloak client credentials **or** a static inference API key:
   - `BASE_URL=https://your-gateway.example.com`
   - `KEYCLOAK_REALM=master`
   - `KEYCLOAK_CLIENT_ID=api`
   - `KEYCLOAK_CLIENT_SECRET=your_client_secret`
   - _or_ `INFERENCE_API_KEY=your_gateway_token`
3. Restart the server.

#### "No documents uploaded"

**Solution**:

- Upload a PDF first using the `/upload-pdf` endpoint
- Check server logs for any upload errors
- Verify the PDF is not corrupted or empty

#### "Could not load vector store"

**Solution**:

- The vector store is created when you upload your first PDF
- Check that the application has write permissions in the directory
- Verify `dmv_index/` directory exists and is accessible

#### Import errors

**Solution**:

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Verify you're using Python 3.10 or higher: `python --version`
3. Activate your virtual environment if using one

#### Server won't start

**Solution**:

1. Check if port 5001 is already in use: `lsof -i :5001` (Unix) or `netstat -ano | findstr :5001` (Windows)
2. Use a different port if needed: `uvicorn server:app --port 5002`
3. Check the logs for specific error messages

#### PDF upload fails

**Solution**:

1. Verify the file is a valid PDF
2. Check file size (must be under 50MB by default)
3. Ensure the PDF contains extractable text (not just images)
4. Check server logs for detailed error messages

#### Query returns no answer

**Solution**:

1. Verify a document has been uploaded successfully
2. Try rephrasing your question
3. Check if the document contains relevant information
4. Increase `TOP_K_DOCUMENTS` in `config.py` for broader search

## UI Common Issues

### API Connection Issues

**Problem**: "Failed to upload PDF" or "Failed to get response"

**Solution**:

1. Ensure the API server is running on `http://localhost:5001`
2. Check browser console for detailed errors
3. Verify CORS is enabled in the API

### Build Issues

**Problem**: Build fails with dependency errors

**Solution**:

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Styling Issues

**Problem**: Styles not applying

**Solution**:

```bash
# Rebuild Tailwind CSS
npm run dev
```

import axios from 'axios';

// Define the API base URL, which could be configured in an environment variable
const BASE_URL = process.env.VUE_APP_BACKEND_URL;

// Create a function to send the file and confidence data to the backend
export const detectLungCancer = async (file, confidence) => {
  const formData = new FormData();
  formData.append('file', file);          // Add the file (image) to the FormData
  formData.append('confidence', confidence); // Add the confidence value to the FormData

  try {
    const response = await axios.post(`${BASE_URL}detect/`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;  // Return the response from FastAPI backend
  } catch (error) {
    console.error('Error during detection:', error);
    throw error;  // Throw the error to be handled later
  }
};

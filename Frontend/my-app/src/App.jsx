import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [detectImage, setDetectImage] = useState(null);
  const [referenceImages, setReferenceImages] = useState([]);
  const [resultImage, setResultImage] = useState(null);
  const [recognizedFaces, setRecognizedFaces] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle detection image upload
  const handleDetectImageChange = (e) => {
    setDetectImage(e.target.files[0]);
    setResultImage(null);
    setRecognizedFaces([]);
    setError(null);
  };

  // Handle reference images upload
  const handleReferenceImagesChange = (e) => {
    setReferenceImages([...e.target.files]);
  };

  // Submit images to FastAPI backend
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!detectImage) {
      setError('Please upload an image for face detection.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', detectImage);

    try {
      // Step 1: Upload detection image and get results
      const response = await axios.post('http://127.0.0.1:8000/recognize_faces/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const { recognized_faces, image_url } = response.data;
      setRecognizedFaces(recognized_faces);
      setResultImage(image_url);

      // Step 2: Optionally upload reference images (if provided)
      if (referenceImages.length > 0) {
        const refFormData = new FormData();
        referenceImages.forEach((file) => refFormData.append('files', file));
        // Assuming you add an endpoint like '/upload_reference_faces/' in FastAPI
        await axios.post('http://127.0.0.1:8000/upload_reference_faces/', refFormData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      }

    } catch (err) {
      setError('Error processing the request: ' + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Face Recognition App</h1>

      {/* Form for uploading images */}
      <form onSubmit={handleSubmit}>
        <div>
          <label>Upload Image for Face Detection:</label>
          <input
            type="file"
            accept="image/*"
            onChange={handleDetectImageChange}
          />
        </div>
        <div>
          <label>Upload Reference Face Images (optional):</label>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={handleReferenceImagesChange}
          />
        </div>
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Recognize Faces'}
        </button>
      </form>

      {/* Display error if any */}
      {error && <p className="error">{error}</p>}

      {/* Display results */}
      {resultImage && (
        <div className="results">
          <h2>Results</h2>
          <img 
  src={resultImage} 
  alt="Processed" 
  className="result-image" 
  style={{ width: "700px", height: "400px" }} 
/>

          <ul>
            {recognizedFaces.map((face, index) => (
              <li key={index}>
                Name: {face.name}, Confidence: {face.confidence}%, Box: [{face.box.join(', ')}]
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
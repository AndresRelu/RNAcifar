import React, { useState } from 'react';
import axios from 'axios';

// Configurar la URL del backend según el entorno
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  // Estados para el botón verde (imágenes del dataset)
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Estados para el botón naranja (imágenes externas)
  const [externalFile, setExternalFile] = useState(null);
  const [externalPreview, setExternalPreview] = useState(null);
  const [externalLoading, setExternalLoading] = useState(false);
  const [externalResult, setExternalResult] = useState(null);
  const [externalError, setExternalError] = useState(null);

  const CLASS_NAMES = ['airplane', 'automobile', 'ship', 'truck'];

  // Manejador para el botón verde (dataset)
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  // Manejador para el botón naranja (imágenes externas)
  const handleExternalFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setExternalFile(file);
      setExternalPreview(URL.createObjectURL(file));
      setExternalResult(null);
      setExternalError(null);
    }
  };

  // Clasificar imágenes del dataset
  const handleClassify = async () => {
    if (!selectedFile) {
      setError('Por favor selecciona una imagen');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('Backend response:', response.data);
      setResult(response.data);
    } catch (err) {
      setError('Error al clasificar la imagen: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Clasificar imágenes externas con procesamiento
  const handleExternalClassify = async () => {
    if (!externalFile) {
      setExternalError('Por favor selecciona una imagen');
      return;
    }

    setExternalLoading(true);
    setExternalError(null);

    const formData = new FormData();
    formData.append('file', externalFile);

    try {
      const response = await axios.post(`${API_URL}/predict-external`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('Backend response (external):', response.data);
      setExternalResult(response.data);
    } catch (err) {
      setExternalError('Error al clasificar la imagen: ' + err.message);
    } finally {
      setExternalLoading(false);
    }
  };

  const getTop3 = () => {
    if (!result || !result.all_probabilities) {
      console.log('No result or all_probabilities:', result);
      return [];
    }
    
    console.log('Result received:', result);
    
    const probs = result.all_probabilities.map((prob, idx) => ({
      class_name: CLASS_NAMES[idx],
      probability: prob,
    }));
    
    return probs.sort((a, b) => b.probability - a.probability).slice(0, 3);
  };

  const getTop3External = () => {
    if (!externalResult || !externalResult.all_probabilities) {
      console.log('No external result or all_probabilities:', externalResult);
      return [];
    }
    
    console.log('External result received:', externalResult);
    
    const probs = externalResult.all_probabilities.map((prob, idx) => ({
      class_name: CLASS_NAMES[idx],
      probability: prob,
    }));
    
    return probs.sort((a, b) => b.probability - a.probability).slice(0, 3);
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>CIFAR-10 Classifier</h1>
      <p style={styles.subtitle}>Clasifica imágenes: airplane, automobile, ship, truck</p>

      {/* SECCIÓN 1: Botón verde para imágenes del dataset */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>📁 Imágenes del Dataset</h2>
        <div style={styles.uploadSection}>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            style={styles.fileInput}
            id="file-input"
          />
          <label htmlFor="file-input" style={styles.uploadButton}>
            Seleccionar Imagen del Dataset
          </label>
        </div>

        {preview && (
          <div style={styles.previewSection}>
            <img src={preview} alt="Preview" style={styles.previewImage} />
          </div>
        )}

        {selectedFile && (
          <button
            onClick={handleClassify}
            disabled={loading}
            style={loading ? styles.buttonDisabled : styles.classifyButton}
          >
            {loading ? 'Clasificando...' : 'Clasificar'}
          </button>
        )}

        {error && (
          <div style={styles.error}>
            {error}
          </div>
        )}

        {result && (
          <div style={styles.resultSection}>
            <h2 style={styles.resultTitle}>Resultado</h2>
            <div style={styles.mainResult}>
              <p style={styles.className}>{result.class_name || 'Desconocido'}</p>
              <p style={styles.confidence}>
                Confianza: {result.confidence != null ? (result.confidence * 100).toFixed(2) : 'N/A'}%
              </p>
            </div>

            <h3 style={styles.top3Title}>Top 3 Predicciones:</h3>
            <div style={styles.top3Container}>
              {getTop3().length > 0 ? (
                getTop3().map((item, idx) => (
                  <div key={idx} style={styles.top3Item}>
                    <span style={styles.top3Class}>{item.class_name}</span>
                    <span style={styles.top3Prob}>
                      {(item.probability * 100).toFixed(2)}%
                    </span>
                  </div>
                ))
              ) : (
                <p>No hay datos de probabilidades disponibles</p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* SEPARADOR */}
      <div style={styles.separator}></div>

      {/* SECCIÓN 2: Botón naranja para imágenes externas */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>🌐 Imagen Externa (Cualquier tamaño)</h2>
        <p style={styles.sectionDescription}>
          Sube cualquier imagen y será automáticamente procesada a 32x32 píxeles
        </p>
        <div style={styles.uploadSection}>
          <input
            type="file"
            accept="image/*"
            onChange={handleExternalFileSelect}
            style={styles.fileInput}
            id="external-file-input"
          />
          <label htmlFor="external-file-input" style={styles.uploadButtonExternal}>
            Seleccionar Imagen Externa
          </label>
        </div>

        {externalPreview && (
          <div style={styles.previewSection}>
            <img src={externalPreview} alt="External Preview" style={styles.previewImage} />
          </div>
        )}

        {externalFile && (
          <button
            onClick={handleExternalClassify}
            disabled={externalLoading}
            style={externalLoading ? styles.buttonDisabled : styles.classifyButtonExternal}
          >
            {externalLoading ? 'Procesando y Clasificando...' : 'Procesar y Clasificar'}
          </button>
        )}

        {externalError && (
          <div style={styles.error}>
            {externalError}
          </div>
        )}

        {externalResult && (
          <div style={styles.resultSection}>
            <h2 style={styles.resultTitle}>Resultado (Procesada a 32x32)</h2>
            <div style={styles.mainResult}>
              <p style={styles.className}>{externalResult.class_name || 'Desconocido'}</p>
              <p style={styles.confidence}>
                Confianza: {externalResult.confidence != null ? (externalResult.confidence * 100).toFixed(2) : 'N/A'}%
              </p>
            </div>

            <h3 style={styles.top3Title}>Top 3 Predicciones:</h3>
            <div style={styles.top3Container}>
              {getTop3External().length > 0 ? (
                getTop3External().map((item, idx) => (
                  <div key={idx} style={styles.top3Item}>
                    <span style={styles.top3Class}>{item.class_name}</span>
                    <span style={styles.top3Prob}>
                      {(item.probability * 100).toFixed(2)}%
                    </span>
                  </div>
                ))
              ) : (
                <p>No hay datos de probabilidades disponibles</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '800px',
    margin: '50px auto',
    padding: '20px',
    fontFamily: 'Arial, sans-serif',
    textAlign: 'center',
  },
  title: {
    color: '#333',
    marginBottom: '10px',
  },
  subtitle: {
    color: '#666',
    marginBottom: '30px',
  },
  section: {
    marginBottom: '30px',
    padding: '20px',
    border: '2px solid #e0e0e0',
    borderRadius: '12px',
    backgroundColor: '#fafafa',
  },
  sectionTitle: {
    color: '#333',
    marginBottom: '10px',
    fontSize: '20px',
  },
  sectionDescription: {
    color: '#777',
    fontSize: '14px',
    marginBottom: '15px',
  },
  separator: {
    height: '2px',
    backgroundColor: '#ddd',
    margin: '40px 0',
  },
  uploadSection: {
    marginBottom: '20px',
  },
  fileInput: {
    display: 'none',
  },
  uploadButton: {
    display: 'inline-block',
    padding: '12px 24px',
    backgroundColor: '#4CAF50',
    color: 'white',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 'bold',
  },
  uploadButtonExternal: {
    display: 'inline-block',
    padding: '12px 24px',
    backgroundColor: '#FF9800',
    color: 'white',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 'bold',
  },
  previewSection: {
    marginBottom: '20px',
  },
  previewImage: {
    maxWidth: '300px',
    maxHeight: '300px',
    border: '2px solid #ddd',
    borderRadius: '8px',
  },
  classifyButton: {
    padding: '12px 48px',
    backgroundColor: '#2196F3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 'bold',
  },
  classifyButtonExternal: {
    padding: '12px 48px',
    backgroundColor: '#F57C00',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 'bold',
  },
  buttonDisabled: {
    padding: '12px 48px',
    backgroundColor: '#ccc',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'not-allowed',
    fontSize: '16px',
    fontWeight: 'bold',
  },
  error: {
    marginTop: '20px',
    padding: '15px',
    backgroundColor: '#ffebee',
    color: '#c62828',
    borderRadius: '4px',
  },
  resultSection: {
    marginTop: '30px',
    padding: '20px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
  },
  resultTitle: {
    color: '#333',
    marginBottom: '15px',
  },
  mainResult: {
    marginBottom: '20px',
  },
  className: {
    fontSize: '32px',
    fontWeight: 'bold',
    color: '#2196F3',
    margin: '10px 0',
    textTransform: 'uppercase',
  },
  confidence: {
    fontSize: '18px',
    color: '#666',
  },
  top3Title: {
    color: '#333',
    marginBottom: '10px',
    fontSize: '18px',
  },
  top3Container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  top3Item: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '10px',
    backgroundColor: 'white',
    borderRadius: '4px',
  },
  top3Class: {
    fontWeight: 'bold',
    color: '#333',
  },
  top3Prob: {
    color: '#666',
  },
};

export default App;

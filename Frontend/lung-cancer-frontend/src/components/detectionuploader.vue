import { detectLungCancer } from '../services/api';

<template>
  <div>
    <h1>Lung Cancer Detection</h1>

    <input type="file" @change="handleFileUpload" />
    <button @click="uploadAndDetect">Upload & Detect</button>

    <div v-if="detections">
      <h3>Detection Results:</h3>
      <p>Detections: {{ detections.detections }}</p>
      <p>Confidence Scores: {{ detections.confidence_scores.map(score => (score * 100).toFixed(2) + '%').join(', ') }} </p>

      <img 
        v-if="detections.image"
        :src="'data:image/jpeg;base64,' + detections.image"
        alt="Detected Image"
        style="display: block; max-width: 100%; height: auto; margin: 10px auto; border: 2px solid white;"
      />
    </div>
  </div>
</template>


<script>
export default {
  data() {
    return {
      file: null,
      detections: null
    };
  },
  methods: {
    handleFileUpload(event) {
      this.file = event.target.files[0];
    },
    async uploadAndDetect() {
      if (!this.file) {
        alert("Please select a file first.");
        return;
      }

      let formData = new FormData();
      formData.append("file", this.file);
      formData.append("confidence", "0.5");

      try {
        const response = await fetch("http://127.0.0.1:8000/detect/", {
          method: "POST",
          body: formData
        });

        if (!response.ok) throw new Error(`HTTP Error! Status: ${response.status}`);

        const result = await response.json();
        console.log("API Response:", result);  // Debugging log

        this.detections = result;
      } catch (error) {
        console.error("Error uploading file:", error);
      }
    }
  }
};
</script>

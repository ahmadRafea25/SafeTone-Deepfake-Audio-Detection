# SAFE TONE - AI-Powered Deepfake Audio Detection & Speaker Verification Platform

A full-stack web application that leverages artificial intelligence to detect AI-generated (deepfake) audio and verify speaker identity through advanced audio analysis.

## Features

- **Audio Upload & Recording**: Upload WAV, MP3, or M4A files, or record audio directly through the browser
- **Deepfake Detection**: AI-powered analysis to detect if audio is authentic or AI-generated
- **Speaker Verification**: Verify speaker identity with confidence scores
- **Real-time Waveform Visualization**: View audio waveforms during recording and playback
- **Analysis Reports**: Store and download detailed JSON reports of audio analyses
- **User Authentication**: Secure sign-up and login with Supabase Auth
- **Report History**: Access and manage all previous audio analysis reports
- **Beautiful UI**: Modern glassmorphism design with animated backgrounds and smooth transitions

## Tech Stack

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Icon library
- **Vite** - Build tool and dev server

### Backend & Database
- **Supabase** - PostgreSQL database, authentication, and real-time features
- **Supabase Edge Functions** - Serverless API endpoints
- **Supabase Storage** - Audio file storage

### AI & Analysis
- **Mock AI Models** - Simulated deepfake detection and speaker verification scores (ready for real model integration)

## Project Structure

```
src/
├── components/          # Reusable React components
│   ├── AnimatedBackground.tsx
│   ├── Hero.tsx
│   ├── AudioUploader.tsx
│   ├── AnalysisResults.tsx
│   ├── ReportTools.tsx
│   └── AuthModal.tsx
├── contexts/           # React Context providers
│   └── AuthContext.tsx
├── pages/              # Page components
│   ├── AnalyzerPage.tsx
│   └── ReportsPage.tsx
├── lib/                # Utility libraries
│   └── supabase.ts
├── App.tsx             # Main app component
├── main.tsx            # Entry point
├── index.css           # Global styles and animations
└── vite-env.d.ts       # Vite type definitions
```

## Getting Started

### Prerequisites
- Node.js 16+
- npm or yarn
- Supabase account (free tier available at https://supabase.com)

### 1. Setup Supabase

1. Create a Supabase project at https://supabase.com
2. Note your project URL and anon key
3. The database schema and Edge Functions are already configured

### 2. Environment Variables

Copy `.env.example` to `.env` and fill in your Supabase credentials:

```bash
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 3. Install Dependencies

```bash
npm install
```

### 4. Development

Run the development server:

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### 5. Production Build

Build for production:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

## Usage

1. **Sign Up**: Create a new account or sign in with existing credentials
2. **Upload/Record Audio**: Choose to upload an audio file or record from your microphone
3. **Analyze**: Click "Analyze Audio" to process the file
4. **View Results**: See authenticity and speaker verification scores with visual indicators
5. **Download Report**: Export analysis results as JSON
6. **View History**: Access all previous reports in the Reports section

## API Endpoints

### Analyze Audio
- **Endpoint**: `/functions/v1/analyze-audio`
- **Method**: POST
- **Auth**: Bearer token (JWT)
- **Body**: FormData with audio file
- **Response**:
```json
{
  "success": true,
  "result": {
    "authenticityScore": 87,
    "authenticityLabel": "Authentic",
    "speakerScore": 82,
    "speakerLabel": "Match"
  },
  "reportId": "uuid"
}
```

## Database Schema

### `audio_reports`
Stores analysis results for each audio file processed

### `user_profiles`
Stores user profile information linked to auth.users

### `audio-files` (Storage)
Bucket for storing uploaded and recorded audio files

## Features & Capabilities

### Audio Analysis
- Deepfake detection with authenticity scoring (0-100%)
- Speaker identity verification with match scoring (0-100%)
- AI-generated audio detection
- Real-time audio waveform visualization during recording

### User Experience
- Responsive design for desktop and mobile
- Animated gradient backgrounds with particle effects
- Glassmorphism UI components with hover effects
- Smooth animations and transitions
- Real-time form validation
- Error handling and user feedback

### Data Management
- Secure user authentication with Supabase Auth
- Row-Level Security (RLS) policies for data protection
- Report history with full CRUD operations
- JSON report export functionality

## Future Enhancements

### Real AI Model Integration
Replace mock scoring with actual ML models:
- **Deepfake Detection**: Integrate Wav2Vec2 + LSTM classifier or Resemblyzer
- **Speaker Verification**: Use ECAPA-TDNN embeddings or speaker_recognition library
- **Whisper Integration**: Add speech-to-text and language identification

### Additional Features
- Reference voice profile management
- Batch audio analysis
- Advanced filtering and search in reports
- Report sharing and collaboration
- Multi-language support
- Dark/light theme toggle
- API key management for programmatic access
- Webhooks for third-party integrations

### Performance Optimization
- Add caching for frequently analyzed audio
- Implement background job processing
- Add analytics and usage metrics
- Optimize model inference latency

## Security Considerations

- All audio files are encrypted in transit and at rest
- Row-Level Security policies ensure users can only access their own data
- Authentication tokens are securely managed by Supabase
- No sensitive data is logged or exposed
- CORS headers properly configured on Edge Functions

## Troubleshooting

### Microphone Access Denied
- Check browser permissions for microphone access
- Ensure HTTPS is used in production (required for getUserMedia)

### Audio Upload Fails
- Verify file format is WAV, MP3, or M4A
- Check file size limits (typically 100MB max)
- Ensure stable internet connection

### Authentication Issues
- Clear browser cookies and local storage
- Verify Supabase URL and keys are correct
- Check that email confirmation is disabled in Supabase (default)

## License

MIT License - feel free to use this project for personal or commercial purposes

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Supabase documentation: https://supabase.com/docs
3. Check browser console for detailed error messages

---

Built with React, TypeScript, Tailwind CSS, and Supabase

# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2024-12-19

### 🚀 Added
- **Debug Mode Toggle**: Added checkbox in sidebar to control App Store review collection debugging
- **Multiple Date Format Support**: App Store now handles various date formats (UTC, timezone, no seconds, date only)
- **Alternative RSS Methods**: Fallback methods for App Store review collection when primary method fails

### 🔧 Changed
- **Complete Card Redesign**: Replaced HTML/CSS cards with native Streamlit components
- **Better Error Handling**: Improved error messages and debugging information
- **Enhanced App Store Integration**: More robust review collection with better parsing

### 🐛 Fixed
- **App Store Date Parsing**: Fixed errors with different date formats (e.g., `2025-06-02T09:44:08-07:00`)
- **Card Display Issues**: Resolved problems where HTML code was showing instead of styled cards
- **Search Functionality**: Improved app search and result filtering
- **Streamlit Compatibility**: Better compatibility with Streamlit Cloud deployment

### 🎨 UI/UX Improvements
- **Modern Card Design**: Clean, professional appearance using native Streamlit components
- **Better Visual Hierarchy**: Improved information organization and readability
- **Consistent Styling**: Unified design language across all components
- **Responsive Layout**: Better mobile and desktop experience

### 📱 Technical Improvements
- **Performance**: Better performance without HTML rendering overhead
- **Reliability**: More stable card rendering and interaction
- **Maintainability**: Cleaner code structure using Streamlit best practices
- **Accessibility**: Better screen reader support and keyboard navigation

## [1.1.0] - 2024-12-18

### 🚀 Added
- **Advanced Text Analysis**: Integrated NLTK for sophisticated text processing
- **App Store Reviews**: Restored App Store review collection functionality
- **Enhanced Search**: Improved app search with better result filtering

### 🔧 Changed
- **NLP Engine**: Replaced spaCy with NLTK for better compatibility
- **Search Algorithm**: Enhanced fuzzy matching for better app discovery

### 🐛 Fixed
- **Streamlit Cloud Issues**: Resolved deployment problems
- **Dependency Conflicts**: Fixed package version compatibility issues

## [1.0.0] - 2024-12-17

### 🚀 Initial Release
- **Google Play Reviews**: Collect and analyze Google Play app reviews
- **App Store Reviews**: Collect and analyze App Store app reviews
- **AI Analysis**: OpenAI GPT-4 powered review analysis
- **Search Functionality**: Find apps by name across platforms
- **Review Filtering**: Date range and platform filtering
- **Data Visualization**: Charts and statistics for review analysis

---

## Versioning

This project uses [Semantic Versioning](http://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for added functionality in a backwards compatible manner
- **PATCH** version for backwards compatible bug fixes

## Release Process

1. **Development**: Features and fixes developed in `main` branch
2. **Testing**: Local testing and validation
3. **Tagging**: Create annotated tag with version number
4. **Deployment**: Push to GitHub and deploy to Streamlit Cloud
5. **Documentation**: Update CHANGELOG.md and README.md

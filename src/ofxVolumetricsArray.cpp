#include "ofxVolumetricsArray.h"

#define STRINGIFY(A) #A

//--------------------------------------------------------------
ofxVolumetricsArray::ofxVolumetricsArray()
	: ofxVolumetrics()
{
	// Have to call this in subclass.
	setupShader();
}

//--------------------------------------------------------------
void ofxVolumetricsArray::setupShader()
{
    string vertexShader =
#include "shaders/gl3/vert_array.glsl"
    string fragmentShader =
#include "shaders/gl3/frag_array.glsl"

    volumeShader.unload();
    volumeShader.setupShaderFromSource(GL_VERTEX_SHADER, vertexShader);
    volumeShader.setupShaderFromSource(GL_FRAGMENT_SHADER, fragmentShader);
    volumeShader.linkProgram();
}

//--------------------------------------------------------------
void ofxVolumetricsArray::setup(int w, int h, int d, ofVec3f voxelSize, bool usePowerOfTwoTexSize)
{
    volTexWidth = volWidth = renderWidth = w;
    volTexHeight = volHeight = renderHeight = h;
    volTexDepth = volDepth = d;

    if (bIsPowerOfTwo)
	{
        volTexWidth = ofNextPow2(volTexWidth);
        volTexHeight = ofNextPow2(volTexHeight);
        volTexDepth = ofNextPow2(volTexDepth);

        ofLogVerbose("ofxVolumetricsArray::setup") << "Using power of two texture size. Requested: " << w << "x" << h << "x" << d << ". Actual: " << volTexWidth << "x" << volTexHeight << "x" << volTexDepth;
    }

    if (bOwnsTexture && volumeTexture) 
	{
        delete volumeTexture;
    }

    volumeTexture = new ofxTextureArray();
    volumeTexture->allocate(w, h, d, GL_RGBA8);
    bOwnsTexture = true;

    if (bIsPowerOfTwo) 
	{
        // If using cropped power of two, blank out the extra memory.
        unsigned char * d;
        d = new unsigned char[volTexWidth*volTexHeight*volTexDepth*4];
        memset(d, 0, volTexWidth*volTexHeight*volTexDepth*4);
        volumeTexture->loadData(d, volTexWidth, volTexHeight, volTexDepth, 0,0,0, GL_RGBA);

        // Free the memory used to blank the texture.
        delete [] d;
    }

    voxelRatio = voxelSize;
    fboRender.allocate(volTexWidth, volTexHeight, GL_RGBA);
    bIsInitialized = true;
}

//--------------------------------------------------------------
void ofxVolumetricsArray::setup(ofxTextureArray *texture, ofVec3f voxelSize)
{
    if (bOwnsTexture && volumeTexture) 
	{
        delete volumeTexture;
    }

    volumeTexture = texture;
    bOwnsTexture = false;

    volTexWidth = volWidth = renderWidth = volumeTexture->texData.width;
    volTexHeight = volHeight = renderHeight = volumeTexture->texData.height;
    volTexDepth = volDepth = volumeTexture->texData.depth;

    voxelRatio = voxelSize;

    if (fboRender.getWidth() != volTexWidth || fboRender.getHeight() != volTexHeight) 
	{
        fboRender.allocate(volTexWidth, volTexHeight, GL_RGBA);
    }
    bIsInitialized = true;
}

//--------------------------------------------------------------
void ofxVolumetricsArray::updateShaderUniforms(int zOffset)
{
	ofxVolumetrics::updateShaderUniforms(zOffset);

	volumeShader.setUniformTexture("volume_tex", volumeTexture->texData.textureTarget, volumeTexture->texData.textureID, 1);
}

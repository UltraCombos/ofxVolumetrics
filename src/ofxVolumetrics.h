#pragma once

/* ofxVolumetrics - render volumetric data on the GPU

 Written by Timothy Scaffidi (http://timothyscaffidi.com)
 Volumetric rendering algorithm adapted from Peter Trier (http://www.daimi.au.dk/~trier/?page_id=98)

*/
#include "ofMain.h"
#include "ofxTexture.h"

class ofxVolumetrics
{
public:
	ofxVolumetrics();
	virtual ~ofxVolumetrics();

	virtual void setup(int w, int h, int d, ofVec3f voxelSize, bool usePowerOfTwoTexSize = false) = 0;
	void setup(ofxTexture *texture, ofVec3f voxelSize);

	void clear();

	virtual void updateVolumeData(unsigned char * data, int w, int h, int d, int xOffset, int yOffset, int zOffset);
	
	void drawVolume(float x, float y, float z, float size, int zTexOffset);
	void drawVolume(float x, float y, float z, float w, float h, float d, int zTexOffset);
	
	bool isInitialized();
	
	int getVolumeWidth();
	int getVolumeHeight();
	int getVolumeDepth();
	
	ofFbo & getFboReference();
	int getRenderWidth();
	int getRenderHeight();
	
	float getXyQuality();
	float getZQuality();
	float getThreshold();
	float getDensity();
	
	void setXyQuality(float q);
	void setZQuality(float q);
	void setThreshold(float t);
	void setDensity(float d);

	void setRenderSettings(float xyQuality, float zQuality, float dens, float thresh);
	void setVolumeTextureFilterMode(GLint filterMode);

protected:
	void setupVbo();
	virtual void setupShader() = 0;
	virtual void updateShaderUniforms(int zOffset);

	void drawRGBCube();
	void updateRenderDimensions();

	ofFbo fboRender;
	ofShader volumeShader;

	ofxTexture *volumeTexture;
	bool bOwnsTexture;

	ofVbo volVbo;

	ofVec3f voxelRatio;
	bool bIsInitialized;
	int volWidth, volHeight, volDepth;
	int volTexWidth, volTexHeight, volTexDepth;
	bool bIsPowerOfTwo;
	ofVec3f quality;
	float threshold;
	float density;
	int renderWidth, renderHeight;
};

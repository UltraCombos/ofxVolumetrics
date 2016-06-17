#include "ofxTexture.h"
#include "ofTexture.cpp"

//----------------------------------------------------------
ofxTexture::ofxTexture()
{}

//----------------------------------------------------------
ofxTexture::~ofxTexture()
{
	clear();
}

//----------------------------------------------------------
void ofxTexture::ofRetain()
{
	retain(texData.textureID);
}

//----------------------------------------------------------
void ofxTexture::ofRelease()
{
	release(texData.textureID);
}

//----------------------------------------------------------
void ofxTexture::clear()
{
	ofRelease();
	texData.textureID = 0;
	texData.bAllocated = false;
}

//----------------------------------------------------------
void ofxTexture::loadData(unsigned char * data, int w, int h, int d, int xOffset, int yOffset, int zOffset, int glFormat)
{
	loadData((void *)data, w, h, d, xOffset, yOffset, zOffset, glFormat);
}

//----------------------------------------------------------
void ofxTexture::loadData(float* data, int w, int h, int d, int xOffset, int yOffset, int zOffset, int glFormat)
{
	loadData((void *)data, w, h, d, xOffset, yOffset, zOffset, glFormat);
}

//----------------------------------------------------------
void ofxTexture::loadData(unsigned short* data, int w, int h, int d, int xOffset, int yOffset, int zOffset, int glFormat)
{
	loadData((void *)data, w, h, d, xOffset, yOffset, zOffset, glFormat);
}

//----------------------------------------------------------
void ofxTexture::loadData(ofPixels & pix, int d, int xOffset, int yOffset, int zOffset)
{
	loadData(pix.getData(), pix.getWidth(), pix.getHeight(), d, xOffset, yOffset, zOffset, ofGetGlFormat(pix));
}

//----------------------------------------------------------
void ofxTexture::loadData(ofShortPixels & pix, int d, int xOffset, int yOffset, int zOffset)
{
	loadData(pix.getData(), pix.getWidth(), pix.getHeight(), d, xOffset, yOffset, zOffset, ofGetGlFormat(pix));
}

//----------------------------------------------------------
void ofxTexture::loadData(ofFloatPixels & pix, int d, int xOffset, int yOffset, int zOffset)
{
	loadData(pix.getData(), pix.getWidth(), pix.getHeight(), d, xOffset, yOffset, zOffset, ofGetGlFormat(pix));
}

//----------------------------------------------------------
void ofxTexture::bind()
{
	glActiveTexture((GLuint)texData.textureID);
	glBindTexture(texData.textureTarget, (GLuint)texData.textureID);
}

//----------------------------------------------------------
void ofxTexture::unbind()
{
	glActiveTexture(0);
}

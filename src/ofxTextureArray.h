#pragma once

#include "ofxTexture.h"

class ofxTextureArray
	: public ofxTexture
{
public:
	ofxTextureArray();

	void allocate(int w, int h, int d, int internalGlDataType) override;

	void enableMipmap(bool hasMipmap = true, float ignored_alpha = 0.1f);

	using ofxTexture::loadData;
protected:
	void loadData(void * data, int w, int h, int d, int xOffset, int yOffset, int zOffset, int glFormat) override;
private: 
	bool hasMipmap = false;
	float ignored_alpha = 0.5f;
};

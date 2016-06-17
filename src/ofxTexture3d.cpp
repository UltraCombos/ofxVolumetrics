#include "ofxTexture3d.h"

//----------------------------------------------------------
ofxTexture3d::ofxTexture3d()
	: ofxTexture()
{
	texData.textureTarget = GL_TEXTURE_3D;
}

//----------------------------------------------------------
void ofxTexture3d::allocate(int w, int h, int d, int internalGlDataType)
{
	if (ofGetLogLevel() >= OF_LOG_VERBOSE)
	{
		int gl_maxTexSize;
		glGetIntegerv(GL_MAX_TEXTURE_SIZE, &gl_maxTexSize);
		int gl_max3DTexSize;
		glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &gl_max3DTexSize);
		ofLogVerbose("ofxTexture3d::allocate") << "Max size is " << gl_maxTexSize << "x" << gl_maxTexSize << "x" << gl_max3DTexSize;
	}
	
	texData.tex_w = w;
    texData.tex_h = h;
    texData.tex_d = d;
    texData.tex_t = w;
    texData.tex_u = h;
    texData.tex_v = d;
    texData.textureTarget = GL_TEXTURE_3D;

    texData.glInternalFormat = internalGlDataType;
	// Get the glType (format) and pixelType (type) corresponding to the glTypeInteral (internalFormat)
	texData.glType = ofGetGLFormatFromInternal(texData.glInternalFormat);
    texData.pixelType = ofGetGlTypeFromInternal(texData.glInternalFormat);

	// Attempt to free the previous bound texture.
	clear();

	// Initialize the new texture.
	glGenTextures(1, (GLuint *)&texData.textureID);
	ofRetain();

	bind();
    glTexImage3D(texData.textureTarget, 0, texData.glInternalFormat, (GLint)texData.tex_w, (GLint)texData.tex_h, (GLint)texData.tex_d, 0, texData.glType, texData.pixelType, 0);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	unbind();

    texData.width = w;
    texData.height = h;
    texData.depth = d;
    texData.bFlipTexture = false;
    texData.bAllocated = true;
}

//----------------------------------------------------------
void ofxTexture3d::loadData(void * data, int w, int h, int d, int xOffset, int yOffset, int zOffset, int glFormat)
{
    if (glFormat!=texData.glType)
    {
		ofLogError("ofxTexture3d::loadData") << "Failed to upload format " << ofGetGlInternalFormatName(glFormat) << " data to " << ofGetGlInternalFormatName(texData.glType) << " texture";
		return;
    }

    if (w > texData.tex_w || h > texData.tex_h || d > texData.tex_d)
    {
		ofLogError("ofxTexture3d::loadData") << "Failed to upload " << w << "x" << h << "x" << d << " data to " << texData.tex_w << "x" << texData.tex_h << "x" << texData.tex_d << " texture";
		return;
    }

    ofSetPixelStoreiAlignment(GL_UNPACK_ALIGNMENT, w, 1, ofGetNumChannelsFromGLFormat(glFormat));
	bind();
    glTexSubImage3D(texData.textureTarget, 0, xOffset, yOffset, zOffset, w, h, d, texData.glType, texData.pixelType, data);
	unbind();
}

#include "ofxTextureArray.h"
#include "opencv.hpp"
//----------------------------------------------------------
ofxTextureArray::ofxTextureArray()
	: ofxTexture()
{
	texData.textureTarget = GL_TEXTURE_2D_ARRAY;
}

//----------------------------------------------------------
void ofxTextureArray::allocate(int w, int h, int d, int internalGlDataType)
{
	if (ofGetLogLevel() >= OF_LOG_VERBOSE)
	{
		int gl_maxTexSize;
		glGetIntegerv(GL_MAX_TEXTURE_SIZE, &gl_maxTexSize);
		int gl_maxTexLayers;
		glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &gl_maxTexLayers);
		ofLogVerbose("ofxTextureArray::allocate") << "Max size is " << gl_maxTexSize << "x" << gl_maxTexSize << "x" << gl_maxTexLayers;
	}
	
	texData.tex_w = w;
	texData.tex_h = h;
	texData.tex_d = d;
	texData.tex_t = w;
	texData.tex_u = h;
	texData.tex_v = d;
	texData.textureTarget = GL_TEXTURE_2D_ARRAY;

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
	if (hasMipmap)
	{
		int level = std::max<int>(1,(int)log2(min(w, h)));
		glTexStorage3D(texData.textureTarget, level, texData.glInternalFormat, (GLint)texData.tex_w, (GLint)texData.tex_h, (GLint)texData.tex_d);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	}
	else
	{
		glTexStorage3D(texData.textureTarget, 1, texData.glInternalFormat, (GLint)texData.tex_w, (GLint)texData.tex_h, (GLint)texData.tex_d);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	}
		

	ofTexture t;
	t.enableMipmap();

	
	//glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_REPEAT);

	unbind();

	texData.width = w;
	texData.height = h;
	texData.depth = d;
	texData.bFlipTexture = false;
	texData.bAllocated = true;
}
//----------------------------------------------------------
void ofxTextureArray::enableMipmap()
{
	hasMipmap = true;
	texData.minFilter = GL_LINEAR_MIPMAP_LINEAR;
}
namespace
{
	void my_resize(cv::Mat& src, cv::Mat& dst, cv::Size size)
	{
		dst = cv::Mat(size, src.type());
		int max_w = src.cols-1;
		int max_h = src.rows-1;
		float scale_x = (float)src.cols / dst.cols;
		float scale_y = (float)src.rows / dst.rows;
		for (int dy = 0; dy < size.width; ++dy)
		{
			for (int dx = 0; dx < size.height;++dx)
			{

				float sx = src.cols*((0.5f + dx) / dst.cols) - 0.5f;
				float sy = src.rows*((0.5f + dy) / dst.rows) - 0.5f;
				
				float sx0 = ofClamp(floor(sx) + 0, 0, src.cols - 1);
				float sy0 = ofClamp(floor(sy) + 0, 0, src.rows - 1);
				float sx1 = ofClamp(floor(sx) + 1, 0, src.cols - 1);
				float sy1 = ofClamp(floor(sy) + 1, 0, src.rows - 1);
				float alpha = sx - sx0;
				float beta = sy - sy0;

				//a---b
				//|   |
				//c---d
				cv::Vec4b & a = src.at<cv::Vec4b>(sy0,sx0);
				cv::Vec4b & b = src.at<cv::Vec4b>(sy0,sx1);
				cv::Vec4b & c = src.at<cv::Vec4b>(sy1,sx0);
				cv::Vec4b & d = src.at<cv::Vec4b>(sy1,sx1);
				
				float wa = a[3] == 0 ? 0.f : (1 - alpha)*(1 - beta);
				float wb = b[3] == 0 ? 0.f : (    alpha)*(1 - beta);
				float wc = c[3] == 0 ? 0.f : (1 - alpha)*(    beta);
				float wd = d[3] == 0 ? 0.f : (    alpha)*(    beta);
				
				dst.at<cv::Vec4b>(dy, dx) = (a*wa + b*wb + c*wc + d*wd) / (wa + wb + wc + wd);
			}
		}

	}
}
//----------------------------------------------------------
void ofxTextureArray::loadData(void * data, int w, int h, int d, int xOffset, int yOffset, int layerOffset, int glFormat)
{
	if (glFormat != texData.glType)
	{
		ofLogError("ofxTextureArray::loadData") << "Failed to upload format " << ofGetGlInternalFormatName(glFormat) << " data to " << ofGetGlInternalFormatName(texData.glType) << " texture";
		return;
	}

	if (w > texData.tex_w || h > texData.tex_h || d > texData.tex_d)
	{
		ofLogError("ofxTextureArray::loadData") << "Failed to upload " << w << "x" << h << "x" << d << " data to " << texData.tex_w << "x" << texData.tex_h << "x" << texData.tex_d << " texture";
		return;
	}

	ofSetPixelStoreiAlignment(GL_UNPACK_ALIGNMENT, w, 1, ofGetNumChannelsFromGLFormat(glFormat));
	bind();
	glTexSubImage3D(texData.textureTarget, 0, xOffset, yOffset, layerOffset, w, h, d, texData.glType, texData.pixelType, data);

	if (hasMipmap)
	{
#if 1
		int level = std::max<int>(1, (int)log2(min(w, h)));
		int width = w;
		int height = h;
		cv::Mat tmp(height, width, CV_8UC4, data);

		for (int i = 1; i < level; ++i)
		{		

			width  = max(1,width / 2);
			height = max(1,height / 2);
			cv::Mat nxt;
			my_resize(tmp, nxt, cv::Size(width, height));
			tmp = nxt;
			glTexSubImage3D(texData.textureTarget, i, xOffset, yOffset, layerOffset, width, height, d, texData.glType, texData.pixelType, tmp.data);
			
		}
#else
		glGenerateMipmap(texData.textureTarget);
#endif
	}
	unbind();
}


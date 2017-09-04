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
void ofxTextureArray::enableMipmap(bool hasMipmap/* = true*/, float ignored_alpha/* = 0.1f*/)
{
	//hasMipmap = false;
	this->hasMipmap = hasMipmap;
	this->ignored_alpha = ignored_alpha;
	texData.minFilter = hasMipmap ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR;
}

namespace
{
	void alpha_resize(cv::Mat& src, cv::Mat& dst, cv::Size size, float ignored_alpha)
	{
		cv::Mat src32f;
		cv::Mat dst32f;
		int depth = src.depth();
		
		float scale;
		switch (src.depth())
		{
		case CV_8U:
			scale = 255.f;
			break;
		case CV_16U:
			scale = 65535.f;
			break;
		default:
		case CV_32F:
			scale = 1.f;
			break;
		}

		src.convertTo(src32f, CV_32FC4,1.f/ scale);
		dst32f = cv::Mat(size, CV_32FC4);
		//dst = cv::Mat(size, src.type());

		int max_w = src32f.cols-1;
		int max_h = src32f.rows-1;
		float scale_x = (float)src32f.cols / dst32f.cols;
		float scale_y = (float)src32f.rows / dst32f.rows;
		for (int dy = 0; dy < size.width; ++dy)
		{
			for (int dx = 0; dx < size.height;++dx)
			{

				float sx = src32f.cols*((0.5f + dx) / dst32f.cols) - 0.5f;
				float sy = src32f.rows*((0.5f + dy) / dst32f.rows) - 0.5f;
				
				float sx0 = ofClamp(floor(sx) + 0, 0, src32f.cols - 1);
				float sy0 = ofClamp(floor(sy) + 0, 0, src32f.rows - 1);
				float sx1 = ofClamp(floor(sx) + 1, 0, src32f.cols - 1);
				float sy1 = ofClamp(floor(sy) + 1, 0, src32f.rows - 1);
				float a = sx - sx0;
				float b = sy - sy0;				
				//p[0]---p[1]
				// |      |
				//p[2]---p[3]
				cv::Vec4f  p[4] = {
					src32f.at<cv::Vec4f>(sy0,sx0),
					src32f.at<cv::Vec4f>(sy0,sx1),
					src32f.at<cv::Vec4f>(sy1,sx0),
					src32f.at<cv::Vec4f>(sy1,sx1),
				};

				float w[4]=
				{
					(1 - a)*(1 - b),
					(    a)*(1 - b),
					(1 - a)*(    b),
					(    a)*(    b),
				};

				float weight_sum=0.f;
				ofVec4f color;
				for (int i = 0; i < 4; ++i)
				{
					float weight = p[i][3] < ignored_alpha ? 0 : w[i];
					color[0] += p[i][0] * weight;
					color[1] += p[i][1] * weight;
					color[2] += p[i][2] * weight;
					color[3] += p[i][3] * weight;
					weight_sum += weight;					
				}
				
				cv::Vec4f & result = dst32f.at<cv::Vec4f>(dy, dx);
				if (weight_sum == 0)
					result = cv::Vec4f(0, 0, 0, 0);
				else
				{
					for (int ch = 0; ch < 4; ++ch)
						result[ch] = color[ch] / weight_sum;
				}
			}
		}

		dst32f.convertTo(dst, src.type(), scale);
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

	int numChannels = ofGetNumChannelsFromGLFormat(glFormat);
	int bytePerChannel;
	int cvType;
	switch (texData.pixelType)
	{
	case GL_UNSIGNED_BYTE:
		bytePerChannel = 1;
		cvType = CV_8U;
		break;
	case GL_UNSIGNED_SHORT:
		bytePerChannel = 2;
		cvType = CV_16U;
		break;
	case GL_FLOAT:
		cvType = CV_32F;		
		bytePerChannel = 4;
		break;
	default:
		ofLogError("ofxTextureArray::loadData") << "Unsupported format " << ofGetGlInternalFormatName(glFormat);
		return;
	}
	
	ofSetPixelStoreiAlignment(GL_UNPACK_ALIGNMENT, w, bytePerChannel, numChannels);
	bind();
	glTexSubImage3D(texData.textureTarget, 0, xOffset, yOffset, layerOffset, w, h, d, texData.glType, texData.pixelType, data);

	if (hasMipmap)
	{
		if (numChannels == 4)
		{
			int level = std::max<int>(1, (int)log2(min(w, h)));
			int width = w;
			int height = h;
			cv::Mat tmp(height, width, CV_MAKETYPE(cvType, numChannels), data);
			for (int i = 1; i < level; ++i)
			{
				width = max(1, width / 2);
				height = max(1, height / 2);
				cv::Mat nxt;
				alpha_resize(tmp, nxt, cv::Size(width, height),i==1?ignored_alpha:0.f);
				tmp = nxt;
				glTexSubImage3D(texData.textureTarget, i, xOffset, yOffset, layerOffset, width, height, d, texData.glType, texData.pixelType, tmp.data);
			}
		}
		else
		{
			glGenerateMipmap(texData.textureTarget);
		}
	}
	unbind();
}


#include "ofxVolumetrics.h"

//--------------------------------------------------------------
ofxVolumetrics::ofxVolumetrics()
{
    volumeTexture = nullptr;

    quality = ofVec3f(1.0);
    threshold = 1.0/255.0;
    density = 1.0;
    volWidth = renderWidth = 0;
    volHeight = renderHeight = 0;
    volDepth = 0;
    bIsInitialized = false;

    setupVbo();
}

//--------------------------------------------------------------
ofxVolumetrics::~ofxVolumetrics()
{
    clear();
}

//--------------------------------------------------------------
void ofxVolumetrics::setupVbo()
{
    ofVec3f volVerts[24];
    ofVec3f volNormals[24];
    ofIndexType volIndices[36];

    /* Front side */
    volNormals[0] = ofVec3f(0.0, 0.0, 1.0);
    volNormals[1] = ofVec3f(0.0, 0.0, 1.0);
    volNormals[2] = ofVec3f(0.0, 0.0, 1.0);
    volNormals[3] = ofVec3f(0.0, 0.0, 1.0);

    volVerts[0] = ofVec3f(1.0, 1.0, 1.0);
    volVerts[1] = ofVec3f(0.0, 1.0, 1.0);
    volVerts[2] = ofVec3f(0.0, 0.0, 1.0);
    volVerts[3] = ofVec3f(1.0, 0.0, 1.0);

    volIndices[0] = 0;
    volIndices[1] = 1;
    volIndices[2] = 2;

    volIndices[3] = 2;
    volIndices[4] = 3;
    volIndices[5] = 0;

    /* Right side */
    volNormals[4] = ofVec3f(1.0, 0.0, 0.0);
    volNormals[5] = ofVec3f(1.0, 0.0, 0.0);
    volNormals[6] = ofVec3f(1.0, 0.0, 0.0);
    volNormals[7] = ofVec3f(1.0, 0.0, 0.0);

    volVerts[4] = ofVec3f(1.0, 1.0, 1.0);
    volVerts[5] = ofVec3f(1.0, 0.0, 1.0);
    volVerts[6] = ofVec3f(1.0, 0.0, 0.0);
    volVerts[7] = ofVec3f(1.0, 1.0, 0.0);

    volIndices[6] = 4;
    volIndices[7] = 5;
    volIndices[8] = 6;

    volIndices[9] = 6;
    volIndices[10] = 7;
    volIndices[11] = 4;

    /* Top side */
    volNormals[8] = ofVec3f(0.0, 1.0, 0.0);
    volNormals[9] = ofVec3f(0.0, 1.0, 0.0);
    volNormals[10] = ofVec3f(0.0, 1.0, 0.0);
    volNormals[11] = ofVec3f(0.0, 1.0, 0.0);

    volVerts[8] = ofVec3f(1.0, 1.0, 1.0);
    volVerts[9] = ofVec3f(1.0, 1.0, 0.0);
    volVerts[10] = ofVec3f(0.0, 1.0, 0.0);
    volVerts[11] = ofVec3f(0.0, 1.0, 1.0);

    volIndices[12] = 8;
    volIndices[13] = 9;
    volIndices[14] = 10;

    volIndices[15] = 10;
    volIndices[16] = 11;
    volIndices[17] = 8;

    /* Left side */
    volNormals[12] = ofVec3f(-1.0, 0.0, 0.0);
    volNormals[13] = ofVec3f(-1.0, 0.0, 0.0);
    volNormals[14] = ofVec3f(-1.0, 0.0, 0.0);
    volNormals[15] = ofVec3f(-1.0, 0.0, 0.0);

    volVerts[12] = ofVec3f(0.0, 1.0, 1.0);
    volVerts[13] = ofVec3f(0.0, 1.0, 0.0);
    volVerts[14] = ofVec3f(0.0, 0.0, 0.0);
    volVerts[15] = ofVec3f(0.0, 0.0, 1.0);

    volIndices[18] = 12;
    volIndices[19] = 13;
    volIndices[20] = 14;

    volIndices[21] = 14;
    volIndices[22] = 15;
    volIndices[23] = 12;

    /* Bottom side */
    volNormals[16] = ofVec3f(0.0, -1.0, 0.0);
    volNormals[17] = ofVec3f(0.0, -1.0, 0.0);
    volNormals[18] = ofVec3f(0.0, -1.0, 0.0);
    volNormals[19] = ofVec3f(0.0, -1.0, 0.0);

    volVerts[16] = ofVec3f(0.0, 0.0, 0.0);
    volVerts[17] = ofVec3f(1.0, 0.0, 0.0);
    volVerts[18] = ofVec3f(1.0, 0.0, 1.0);
    volVerts[19] = ofVec3f(0.0, 0.0, 1.0);

    volIndices[24] = 16;
    volIndices[25] = 17;
    volIndices[26] = 18;

    volIndices[27] = 18;
    volIndices[28] = 19;
    volIndices[29] = 16;

    /* Back side */
    volNormals[20] = ofVec3f(0.0, 0.0, -1.0);
    volNormals[21] = ofVec3f(0.0, 0.0, -1.0);
    volNormals[22] = ofVec3f(0.0, 0.0, -1.0);
    volNormals[23] = ofVec3f(0.0, 0.0, -1.0);

    volVerts[20] = ofVec3f(1.0, 0.0, 0.0);
    volVerts[21] = ofVec3f(0.0, 0.0, 0.0);
    volVerts[22] = ofVec3f(0.0, 1.0, 0.0);
    volVerts[23] = ofVec3f(1.0, 1.0, 0.0);

    volIndices[30] = 20;
    volIndices[31] = 21;
    volIndices[32] = 22;

    volIndices[33] = 22;
    volIndices[34] = 23;
    volIndices[35] = 20;

    volVbo.setVertexData(volVerts, 24, GL_STATIC_DRAW);
    volVbo.setNormalData(volNormals, 24, GL_STATIC_DRAW);
    volVbo.setAttributeData(ofShader::COLOR_ATTRIBUTE, (float *)volVerts, 3, 24, GL_STATIC_DRAW);
    volVbo.setAttributeData(ofShader::TEXCOORD_ATTRIBUTE, (float *)volVerts, 3, 24, GL_STATIC_DRAW);
    volVbo.setIndexData(volIndices, 36, GL_STATIC_DRAW);
}


void ofxVolumetrics::setup(ofxTexture *texture, ofVec3f voxelSize)
{
    if (bOwnsTexture && volumeTexture) {
        delete volumeTexture;
    }

    volumeTexture = texture;
    bOwnsTexture = false;

    volTexWidth = volWidth = renderWidth = volumeTexture->texData.width;
    volTexHeight = volHeight = renderHeight = volumeTexture->texData.height;
    volTexDepth = volDepth = volumeTexture->texData.depth;

    voxelRatio = voxelSize;

    if (fboRender.getWidth() != volTexWidth || fboRender.getHeight() != volTexHeight) {
        fboRender.allocate(volTexWidth, volTexHeight, GL_RGBA);
    }

    bIsInitialized = true;
}

void ofxVolumetrics::clear()
{
    volumeShader.unload();
//    fboBackground.destroy();
//    fboRender.destroy();

    if (bOwnsTexture) {
        volumeTexture->clear();
        delete volumeTexture;
    }
    volumeTexture = nullptr;

    volWidth = renderWidth = 0;
    volHeight = renderHeight = 0;
    volDepth = 0;
    bIsInitialized = false;
}

void ofxVolumetrics::updateVolumeData(unsigned char * data, int w, int h, int d, int xOffset, int yOffset, int zOffset)
{
    volumeTexture->loadData(data, w, h, d, xOffset, yOffset, zOffset, GL_RGBA);
}

void ofxVolumetrics::updateShaderUniforms(int zOffset)
{
	volumeShader.setUniform3f("vol_d", (float)volWidth, (float)volHeight, (float)volDepth); //dimensions of the volume
	volumeShader.setUniform3f("vol_tex_d", (float)volTexWidth, (float)volTexHeight, (float)volTexDepth); //dimensions of the volume texture

	volumeShader.setUniform1f("zoffset", zOffset); // used for animation so that we dont have to upload the entire volume every time
	volumeShader.setUniform1f("quality", quality.z); // 0 ... 1
	volumeShader.setUniform1f("density", density); // 0 ... 1
	volumeShader.setUniform1f("threshold", threshold);//(float)mouseX/(float)ofGetWidth());
}

void ofxVolumetrics::drawVolume(float x, float y, float z, float size, int zTexOffset)
{
    ofVec3f volumeSize = voxelRatio * ofVec3f(volWidth,volHeight,volDepth);
    float maxDim = max(max(volumeSize.x, volumeSize.y), volumeSize.z);
    volumeSize = volumeSize * size / maxDim;

    drawVolume(x, y, z, volumeSize.x, volumeSize.y, volumeSize.z, zTexOffset);
}

void ofxVolumetrics::drawVolume(float x, float y, float z, float w, float h, float d, int zTexOffset)
{
    updateRenderDimensions();

    ofVec3f cubeSize = ofVec3f(w, h, d);

    //GLfloat modl[16], proj[16];
    //glGetFloatv( GL_MODELVIEW_MATRIX, modl);
    //glGetFloatv(GL_PROJECTION_MATRIX, proj);
    GLint color[4];
    glGetIntegerv(GL_CURRENT_COLOR, color);

    ofMatrix4x4 modlMat = ofGetCurrentMatrix(OF_MATRIX_MODELVIEW);
    ofMatrix4x4 projMat = ofGetCurrentMatrix(OF_MATRIX_PROJECTION);

    ofVec3f scale,t;
    ofQuaternion a,b;
    //ofMatrix4x4(modl).decompose(t, a, scale, b);
    modlMat.decompose(t, a, scale, b);

    GLint cull_mode;
    glGetIntegerv(GL_FRONT_FACE, &cull_mode);
    GLint cull_mode_fbo = (scale.x*scale.y*scale.z) > 0 ? GL_CCW : GL_CW;

    /* raycasting pass */
    fboRender.begin();
    volumeShader.begin();
    ofClear(0,0,0,0);

    //load matricies from outside the FBO
    ofSetMatrixMode(OF_MATRIX_PROJECTION);
    ofLoadMatrix(projMat);
    ofSetMatrixMode(OF_MATRIX_MODELVIEW);
    ofLoadMatrix(modlMat);

    //glMatrixMode(GL_PROJECTION);
    //glLoadMatrixf(proj);
    //glMatrixMode(GL_MODELVIEW);
    //glLoadMatrixf(modl);

    ofTranslate(x-cubeSize.x/2, y-cubeSize.y/2, z-cubeSize.z/2);
    ofScale(cubeSize.x,cubeSize.y,cubeSize.z);

	updateShaderUniforms(zTexOffset);

    glFrontFace(cull_mode_fbo);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    drawRGBCube();
    glDisable(GL_CULL_FACE);
    glFrontFace(cull_mode);

    volumeShader.end();
    fboRender.end();

    ofPushView();

    glColor4iv(color);
    ofSetupScreenOrtho();//ofGetWidth(), ofGetHeight(),OF_ORIENTATION_DEFAULT,false,0,1000);
    fboRender.draw(0,0,ofGetWidth(),ofGetHeight());

    ofPopView();

}

void ofxVolumetrics::drawRGBCube()
{
    //volVbo.bind();

    //glEnableClientState(GL_VERTEX_ARRAY);
    //glEnableClientState(GL_NORMAL_ARRAY);
    //glEnableClientState(GL_COLOR_ARRAY);
    //glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    //glVertexPointer(3, GL_FLOAT, sizeof(ofVec3f), volVerts);
    //glNormalPointer(GL_FLOAT, sizeof(ofVec3f), volNormals);
    //glColorPointer(3,GL_FLOAT, sizeof(ofVec3f), volVerts);
    //glTexCoordPointer(3, GL_FLOAT, sizeof(ofVec3f), volVerts);

    //glDrawArrays(GL_QUADS, 0, 24);
    //glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, volIndices);

    //volVbo.draw(GL_QUADS, 0, 24);
    volVbo.drawElements(GL_TRIANGLES, 36);

    //glDisableClientState(GL_COLOR_ARRAY);
    //glDisableClientState(GL_NORMAL_ARRAY);
    //glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    //glDisableClientState(GL_VERTEX_ARRAY);    

    //volVbo.unbind();
}

void ofxVolumetrics::updateRenderDimensions()
{
//    if((int)(ofGetWidth() * quality.x) != renderWidth)
//    {
//        renderWidth = ofGetWidth()*quality.x;
//        renderHeight = ofGetHeight()*quality.x;
//        fboRender.allocate(renderWidth, renderHeight, GL_RGBA);
//    }
}

void ofxVolumetrics::setXyQuality(float q)
{
    quality.x = MAX(q,0.01);

	updateRenderDimensions();
}
void ofxVolumetrics::setZQuality(float q)
{
    quality.z = MAX(q,0.01);
}
void ofxVolumetrics::setThreshold(float t)
{
    threshold = ofClamp(t,0.0,1.0);
}
void ofxVolumetrics::setDensity(float d)
{
    density = MAX(d,0.0);
}
void ofxVolumetrics::setRenderSettings(float xyQuality, float zQuality, float dens, float thresh)
{
    setXyQuality(xyQuality);
    setZQuality(zQuality);
    setDensity(dens);
    setThreshold(thresh);
}

void ofxVolumetrics::setVolumeTextureFilterMode(GLint filterMode) 
{
    if (filterMode != GL_NEAREST && filterMode != GL_LINEAR) return;

    volumeTexture->bind();
    glTexParameteri(volumeTexture->texData.textureTarget, GL_TEXTURE_MIN_FILTER, filterMode);
    glTexParameteri(volumeTexture->texData.textureTarget, GL_TEXTURE_MAG_FILTER, filterMode);
    volumeTexture->unbind();
}

bool ofxVolumetrics::isInitialized()
{
    return bIsInitialized;
}
int ofxVolumetrics::getVolumeWidth()
{
    return volWidth;
}
int ofxVolumetrics::getVolumeHeight()
{
    return volHeight;
}
int ofxVolumetrics::getVolumeDepth()
{
    return volDepth;
}
int ofxVolumetrics::getRenderWidth()
{
    return renderWidth;
}
int ofxVolumetrics::getRenderHeight()
{
    return renderHeight;
}
float ofxVolumetrics::getXyQuality()
{
    return quality.x;
}
float ofxVolumetrics::getZQuality()
{
    return quality.z;
}
float ofxVolumetrics::getThreshold()
{
    return threshold;
}
float ofxVolumetrics::getDensity()
{
    return density;
}
ofFbo & ofxVolumetrics::getFboReference(){
    return fboRender;
}

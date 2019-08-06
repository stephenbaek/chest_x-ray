# -*- coding: utf-8 -*-
"""
U-Net Implementation
Copyright (c) 2019 Visual Intelligence Laboratory
Licensed under the MIT License (see LICENSE for details)
Written by Stephen Baek
"""
import tensorflow as tf

class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__(name='U-Net')
        
        self.enc_conv11 = self.Conv2D(64, name='encoder1_conv1')
        self.enc_conv12 = self.Conv2D(64, name='encoder1_conv2')
        self.enc_pool1  = self.Pool2D(2,  name='encoder1_pool')
        
        self.enc_conv21 = self.Conv2D(128, name='encoder2_conv1')
        self.enc_conv22 = self.Conv2D(128, name='encoder2_conv2')
        self.enc_pool2  = self.Pool2D(2,  name='encoder2_pool')
        
        self.enc_conv31 = self.Conv2D(256, name='encoder3_conv1')
        self.enc_conv32 = self.Conv2D(256, name='encoder3_conv2')
        self.enc_pool3  = self.Pool2D(2,  name='encoder3_pool')
        
        self.enc_conv41 = self.Conv2D(512, name='encoder4_conv1')
        self.enc_conv42 = self.Conv2D(512, name='encoder4_conv2')
        self.enc_drop4  = self.Dropout(0.5)
        self.enc_pool4  = self.Pool2D(2,  name='encoder4_pool')

        self.enc_conv51 = self.Conv2D(1024, name='encoder5_conv1')
        self.enc_conv52 = self.Conv2D(1024, name='encoder5_conv2')
        self.enc_drop5  = self.Dropout(0.5)
        
        self.dec_up1    = self.Upsample(2)
        self.dec_upconv1= self.Conv2D(512, kernel_size=2)
        self.dec_merge1 = self.Merge()
        self.dec_conv11 = self.Conv2D(512, name='decoder1_conv1')
        self.dec_conv12 = self.Conv2D(512, name='decoder1_conv2')
        
        self.dec_up2    = self.Upsample(2)
        self.dec_upconv2= self.Conv2D(256, kernel_size=2)
        self.dec_merge2 = self.Merge()
        self.dec_conv21 = self.Conv2D(256, name='decoder2_conv1')
        self.dec_conv22 = self.Conv2D(256, name='decoder2_conv2')

        self.dec_up3    = self.Upsample(2)
        self.dec_upconv3= self.Conv2D(128, kernel_size=2)
        self.dec_merge3 = self.Merge()
        self.dec_conv31 = self.Conv2D(128, name='decoder3_conv1')
        self.dec_conv32 = self.Conv2D(128, name='decoder3_conv2')
        
        self.dec_up4    = self.Upsample(2)
        self.dec_upconv4= self.Conv2D(64, kernel_size=2)
        self.dec_merge4 = self.Merge()
        self.dec_conv41 = self.Conv2D(64, name='decoder4_conv1')
        self.dec_conv42 = self.Conv2D(64, name='decoder4_conv2')
                
        self.dec_conv51 = self.Conv2D(2, name='decoder5_conv1')
        self.dec_conv52 = self.Conv2D(1, kernel_size=1, activation='sigmoid', name='decoder5_conv2')
        
        
    def Conv2D(self, channels, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name=None):
        return tf.keras.layers.Conv2D(channels, kernel_size=kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer, name=name)
    
    def Pool2D(self, pool_size=2, name=None):
        return tf.keras.layers.MaxPooling2D(pool_size=pool_size, name=name)
    
    def Dropout(self, rate):
        return tf.keras.layers.Dropout(rate=rate)
    
    def Upsample(self, size=2):
        return tf.keras.layers.UpSampling2D(size=size)
    
    def Merge(self, axis=3):
        return tf.keras.layers.Concatenate(axis=axis)
    
    def call(self, inputs):
        enc1 = self.enc_conv11(inputs)
        enc1 = self.enc_conv12(enc1)
        enc2 = self.enc_pool1(enc1)
        
        enc2 = self.enc_conv21(enc2)
        enc2 = self.enc_conv22(enc2)
        enc3 = self.enc_pool2(enc2)
        
        enc3 = self.enc_conv31(enc3)
        enc3 = self.enc_conv32(enc3)
        enc4 = self.enc_pool3(enc3)
        
        enc4 = self.enc_conv41(enc4)
        enc4 = self.enc_conv42(enc4)
        enc4 = self.enc_drop4(enc4)
        enc5 = self.enc_pool4(enc4)
        
        enc5 = self.enc_conv51(enc5)
        enc5 = self.enc_conv52(enc5)
        enc5 = self.enc_drop5(enc5)
        
        dec1 = self.dec_up1(enc5)
        dec1 = self.dec_upconv1(dec1)
        dec1 = self.dec_merge1([enc4, dec1])
        dec1 = self.dec_conv11(dec1)
        dec2 = self.dec_conv12(dec1)
        
        dec2 = self.dec_up2(dec2)
        dec2 = self.dec_upconv2(dec2)
        dec2 = self.dec_merge2([enc3, dec2])
        dec2 = self.dec_conv21(dec2)
        dec3 = self.dec_conv22(dec2)
        
        dec3 = self.dec_up3(dec3)
        dec3 = self.dec_upconv3(dec3)
        dec3 = self.dec_merge3([enc2, dec3])
        dec3 = self.dec_conv31(dec3)
        dec4 = self.dec_conv32(dec3)
        
        dec4 = self.dec_up4(dec4)
        dec4 = self.dec_upconv4(dec4)
        dec4 = self.dec_merge4([enc1, dec4])
        dec4 = self.dec_conv41(dec4)
        dec5 = self.dec_conv42(dec4)
        
        dec5 = self.dec_conv51(dec5)
        
        return self.dec_conv52(dec5)

#input_size=(256,256,1)
#inputs = tf.keras.Input(input_size)
#conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
#conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
#pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
#conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
#pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
#conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#drop4 = tf.keras.layers.Dropout(0.5)(conv4)
#pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
#
#conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#drop5 = tf.keras.layers.Dropout(0.5)(conv5)
#
#up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
#merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
#conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
#
#up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
#merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
#conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
#
#up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
#merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
#conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#
#up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
#merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
#conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#conv9 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#conv10 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
#
#model = tf.keras.Model(inputs, conv10)
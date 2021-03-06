/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package com.iot.smart.tf;

import android.graphics.Bitmap;
import android.graphics.RectF;

import java.util.List;
public interface Classifier {
  List<Recognition> recognizeImage(Bitmap bitmap);
  void setNumThreads(int num_threads);
  class Recognition {
    private final String id;
    private final String title;
    private final Float confidence;
    private RectF location;

    public Recognition(
            final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
    }
    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }
    public RectF getLocation() {
      return new RectF(location);
    }
    public void setLocation(RectF location) {
      this.location = location;
    }
  }
}

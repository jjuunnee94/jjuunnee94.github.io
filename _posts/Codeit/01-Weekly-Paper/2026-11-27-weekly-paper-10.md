---
title: "Weekly paper #10"
date: 2026-11-27 17:00:00 +0900
categories: [Codeit, '01-Weekly-Paper', 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
tags: [codeit, WeeklyPaper]
math: true
---

🇶 자율주행 차량이 카메라로 도로 이미지를 분석해요. '이 픽셀들이 차인지 도로인지 사람인지' 구분하는 것과, '왼쪽 차와 오른쪽 차를 서로 다른 개체로 구별'하는 것은 다른 문제예요. 이 둘을 각각 어떤 방식으로 해결할 건가요?

'이 픽셀들이 차인지 도로인지 사람인지' 구분하는 문제는 Semantic Segmentation 방식으로 해결합니다.  
FCN이나 U-Net 같은 모델을 활용해 이미지의 모든 픽셀에 클래스 라벨을 부여함으로써, 같은 클래스(차, 도로, 사람 등)에 속한 영역을 픽셀 단위로 동일하게 분류합니다.  

'왼쪽 차와 오른쪽 차를 서로 다른 개체로 구별'하는 문제는 Instance Segmentation 방식으로 해결합니다.  
Mask R-CNN처럼 객체 탐지(Object Detection)가 결합된 모델을 사용해 동일한 '차' 클래스 내에서도 개별 객체의 위치(RoI)를 먼저 탐지하고, 각 인스턴스마다 독립적인 픽셀 단위 마스크를 생성하여 서로 다른 개체로 구분합니다.

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 작성한 위클리페이퍼입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }

package com.example.camx_test.Retrofit

import retrofit2.Retrofit
import retrofit2.converter.scalars.ScalarsConverterFactory

object RetrofitClient {
    private var retrofitClient: Retrofit? = null

    val client:Retrofit
        get(){
            if(retrofitClient == null)
                retrofitClient = Retrofit.Builder()
                    .baseUrl("http://10.0.2.2:5000/")
                    .addConverterFactory(ScalarsConverterFactory.create())
                    .build()
            return retrofitClient!!
        }
}
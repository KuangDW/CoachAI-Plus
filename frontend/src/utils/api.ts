import axios, { AxiosRequestConfig } from "axios";
// axios.defaults.headers.post["Content-Type"] = "application/json";
axios.defaults.xsrfHeaderName = "X-CSRFToken";
axios.defaults.xsrfCookieName = "csrftoken";

/**
 * Wrapper function of axios
 *
 * @param url - The base url
 * @param method - HTTP method
 * @param basseURL - Prefix url
 * @param options - Other axios options
 * @returns - Promise object
 **/
export const ajax = async (
  url: string,
  method: string,
  baseURL?: string,
  options?: AxiosRequestConfig<any>
) => {
  if (options !== undefined) {
    var { params = {}, data = {} } = options;
  } else {
    params = data = null;
  }
  const axiosOptions: AxiosRequestConfig<any> = {
    url: url,
    method: method,
    params: params,
    data: data,
    baseURL: baseURL,
    ...options,
  };
  return axios(axiosOptions)
    .then((res) => res)
    .catch((err) => {
      throw err;
    });
};

/**
 * Wrapper of ajax function, use `/api` as baseURL
 *
 * @param url - Url after `/api`
 * @param method - HTTP method
 * @param options - other axios options
 * @returns - Promise object
 */
export const fetchApi = (
  url: string,
  method: string,
  options?: AxiosRequestConfig<any>
) =>
  ajax(url, method, `${import.meta.env.DEV ? import.meta.env.VITE_API_BASE : ""}/api`, options);

export default ajax;